import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import {
  atomicJson,
  kronosInspectorWindows,
  mergeOverlappingWindows,
  parseKronosForecastArtifact,
  policyCalibrationEpisodeSplit,
  policyId,
  validateDenseForecastCoverage,
  type KronosBotPolicy,
} from "./backtest-kronos.js";
import {
  assertPolicyGate,
  assertValidationReport,
  type PolicyIdentity,
  type ValidationIdentity,
} from "./run-kronos-final-pipeline.js";

const REPO_ROOT = path.resolve(import.meta.dirname, "..");
const VALIDATION_CUTOFF = Date.parse("2024-07-01T00:00:00.000Z");
const SOURCE_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a";
const CLEAN_CHECKPOINT_SHA256 =
  "ed1ba73a21ec027b0a0471724c48c8a259724c679234da3ac3dd8eb3e81bd1db";
const EXCLUSION_PLAN_SHA256 =
  "88c86e472acb12dc5206913f8d3368927068c54f6ef3b40e72a026de1ee0b6b1";
const EXCLUDED_WINDOWS = [
  "fit-full", "fit-1", "fit-2", "fit-3", "fit-4", "latest-3m",
] as const;

interface ModelSpec {
  id: "mini" | "small" | "base";
  repoId: string;
  revision: string;
  tokenizerRepoId: string;
  tokenizerRevision: string;
  publishedMaxContext: number;
}

export const MODEL_SPECS: readonly ModelSpec[] = [
  {
    id: "mini",
    repoId: "NeoQuasar/Kronos-mini",
    revision: "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    tokenizerRepoId: "NeoQuasar/Kronos-Tokenizer-2k",
    tokenizerRevision: "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
    publishedMaxContext: 2_048,
  },
  {
    id: "small",
    repoId: "NeoQuasar/Kronos-small",
    revision: "901c26c1332695a2a8f243eb2f37243a37bea320",
    tokenizerRepoId: "NeoQuasar/Kronos-Tokenizer-base",
    tokenizerRevision: "0e0117387f39004a9016484a186a908917e22426",
    publishedMaxContext: 512,
  },
  {
    id: "base",
    repoId: "NeoQuasar/Kronos-base",
    revision: "2b554741eca47781b64468546e77fef3e85130e6",
    tokenizerRepoId: "NeoQuasar/Kronos-Tokenizer-base",
    tokenizerRevision: "0e0117387f39004a9016484a186a908917e22426",
    publishedMaxContext: 512,
  },
] as const;

const REQUIRED_SNAPSHOTS = new Map([
  ...MODEL_SPECS.map((spec) => [spec.repoId, spec.revision] as const),
  ...MODEL_SPECS.map((spec) => [spec.tokenizerRepoId, spec.tokenizerRevision] as const),
]);

const SNAPSHOT_HASHES = new Map<string, { config: string; weights: string }>([
  ["NeoQuasar/Kronos-mini", {
    config: "70daca2cb11e3a979dd6b8ac12ee08e2aace877acf28f5b8dfb4fe5609736201",
    weights: "a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c",
  }],
  ["NeoQuasar/Kronos-small", {
    config: "5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
    weights: "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
  }],
  ["NeoQuasar/Kronos-base", {
    config: "77ebc3038b647709b92be002f801d72e1a385f4c8c2c5aa1cc6cf21fcfe44eb2",
    weights: "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
  }],
  ["NeoQuasar/Kronos-Tokenizer-2k", {
    config: "0b30a443affb03e05a876a083857de9164f899feb7b4d261da02c485c9a3e3b6",
    weights: "b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717",
  }],
  ["NeoQuasar/Kronos-Tokenizer-base", {
    config: "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
    weights: "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
  }],
]);

type JsonRecord = Record<string, unknown>;

function record(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as JsonRecord;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function integer(value: unknown, label: string): number {
  const parsed = finite(value, label);
  if (!Number.isInteger(parsed)) throw new Error(`${label} must be an integer.`);
  return parsed;
}

function equal(actual: unknown, expected: unknown, label: string): void {
  if (actual !== expected) {
    throw new Error(`${label} must equal ${JSON.stringify(expected)}; got ${JSON.stringify(actual)}.`);
  }
}

function equalSet(actual: readonly string[], expected: readonly string[], label: string): void {
  const left = [...actual].sort();
  const right = [...expected].sort();
  if (left.length !== right.length || left.some((value, index) => value !== right[index])) {
    throw new Error(`${label} does not match the expected set.`);
  }
}

function sha256File(file: string): string {
  const hash = crypto.createHash("sha256");
  const descriptor = fs.openSync(file, "r");
  const buffer = Buffer.allocUnsafe(4 * 1024 * 1024);
  try {
    for (;;) {
      const bytes = fs.readSync(descriptor, buffer, 0, buffer.length, null);
      if (bytes === 0) break;
      hash.update(buffer.subarray(0, bytes));
    }
  } finally {
    fs.closeSync(descriptor);
  }
  return hash.digest("hex");
}

function loadJson(file: string): unknown {
  if (!fs.existsSync(file)) throw new Error(`Missing required artifact: ${file}`);
  return JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
}

function resolveArgument(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return path.resolve(REPO_ROOT, inline.slice(prefix.length));
  const index = process.argv.indexOf(`--${name}`);
  return path.resolve(
    REPO_ROOT,
    index >= 0 && process.argv[index + 1] ? process.argv[index + 1]! : fallback,
  );
}

function requiredMetric(container: JsonRecord, key: string, label: string): number {
  return finite(container[key], `${label}.${key}`);
}

interface BenchmarkModelSummary {
  id: string;
  examples: number;
  candleMseSkillVsPersistence: number;
  candleCorrelation: number;
  closeReturnCorrelation: number;
  horizonReturnIc: number;
  priceSeriesIc: number;
  oracleForwardKl: number;
  samplePathCrps: number;
  repairedQuantileValidOhlcFraction: number;
}

export function auditBenchmarkReport(
  source: unknown,
  options: {
    expectedModels: readonly ModelSpec[];
    expectedWindowIds: readonly string[];
    expectedUniqueOrigins: number;
    expectedMemberships: number;
    expectedStride: number;
    expectedModelIds?: readonly string[];
  },
): { runSignature: string; models: BenchmarkModelSummary[] } {
  const root = record(source, "benchmark report");
  equal(root.version, 2, "benchmark.version");
  equal(
    root.contract,
    "kronos-probabilistic-nonfit-inspector-windows-15x1m-v2",
    "benchmark.contract",
  );
  const runSignature = text(root.runSignature, "benchmark.runSignature");
  if (!/^[a-f0-9]{64}$/.test(runSignature)) {
    throw new Error("benchmark.runSignature must be a SHA-256 value.");
  }
  const sourceIdentity = record(root.kronosSource, "benchmark.kronosSource");
  equal(sourceIdentity.commit, SOURCE_COMMIT, "benchmark.kronosSource.commit");
  const data = record(root.data, "benchmark.data");
  equal(data.windows, 28, "benchmark.data.windows");
  equal(data.horizonCandles, 15, "benchmark.data.horizonCandles");
  equal(data.lookbackCandles, 512, "benchmark.data.lookbackCandles");
  equal(data.originStrideCandles, options.expectedStride, "benchmark.data.originStrideCandles");
  equal(data.uniqueOrigins, options.expectedUniqueOrigins, "benchmark.data.uniqueOrigins");
  equal(
    data.windowOriginMemberships,
    options.expectedMemberships,
    "benchmark.data.windowOriginMemberships",
  );
  equalSet(
    array(data.excludedWindows, "benchmark.data.excludedWindows").map((value, index) =>
      text(value, `benchmark.data.excludedWindows[${index}]`)),
    EXCLUDED_WINDOWS,
    "benchmark.data.excludedWindows",
  );
  const sampling = record(root.sampling, "benchmark.sampling");
  equal(sampling.temperature, 0.8, "benchmark.sampling.temperature");
  equal(sampling.topP, 0.9, "benchmark.sampling.topP");
  equal(sampling.sampleCount, 20, "benchmark.sampling.sampleCount");
  equal(sampling.retainedPaths, true, "benchmark.sampling.retainedPaths");
  equal(sampling.constraintRepair, "KQSP", "benchmark.sampling.constraintRepair");

  const models = array(root.models, "benchmark.models");
  equal(models.length, options.expectedModels.length, "benchmark.models.length");
  const byId = new Map(models.map((value, index) => {
    const model = record(value, `benchmark.models[${index}]`);
    return [text(model.id, `benchmark.models[${index}].id`), model] as const;
  }));
  equal(byId.size, models.length, "benchmark unique model ids");
  const summaries: BenchmarkModelSummary[] = [];
  if (options.expectedModelIds
    && options.expectedModelIds.length !== options.expectedModels.length) {
    throw new Error("expectedModelIds must align one-for-one with expectedModels.");
  }
  for (const [modelIndex, spec] of options.expectedModels.entries()) {
    const reportId = options.expectedModelIds?.[modelIndex] ?? spec.id;
    const model = byId.get(reportId);
    if (!model) throw new Error(`Benchmark is missing model ${reportId}.`);
    const identity = record(model.model, `${reportId}.model`);
    equal(identity.repoId, spec.repoId, `${reportId}.model.repoId`);
    equal(identity.revision, spec.revision, `${reportId}.model.revision`);
    equal(identity.tokenizerRepoId, spec.tokenizerRepoId, `${reportId}.model.tokenizerRepoId`);
    equal(
      identity.tokenizerRevision,
      spec.tokenizerRevision,
      `${reportId}.model.tokenizerRevision`,
    );
    equal(
      identity.publishedMaxContext,
      spec.publishedMaxContext,
      `${reportId}.model.publishedMaxContext`,
    );
    equal(model.primaryEstimator, "ensembleMean", `${reportId}.primaryEstimator`);
    const unique = record(model.uniqueOrigins, `${reportId}.uniqueOrigins`);
    equal(unique.examples, options.expectedUniqueOrigins, `${reportId}.uniqueOrigins.examples`);
    const candle = record(unique.candle, `${reportId}.uniqueOrigins.candle`);
    const closeReturn = record(unique.closeReturn, `${reportId}.uniqueOrigins.closeReturn`);
    const paper = record(unique.paperAligned, `${reportId}.uniqueOrigins.paperAligned`);
    const oracle = record(unique.oracle, `${reportId}.uniqueOrigins.oracle`);
    const probabilistic = record(model.probabilistic, `${reportId}.probabilistic`);
    const probabilisticUnique = record(
      probabilistic.uniqueOrigins,
      `${reportId}.probabilistic.uniqueOrigins`,
    );
    const windowReports = array(model.windows, `${reportId}.windows`);
    equal(windowReports.length, 28, `${reportId}.windows.length`);
    equalSet(
      windowReports.map((value, index) =>
        text(record(value, `${reportId}.windows[${index}]`).id, `${reportId}.windows[${index}].id`)),
      options.expectedWindowIds,
      `${reportId}.windowIds`,
    );
    const repaired = requiredMetric(
      probabilisticUnique,
      "repairedQuantileValidOhlcFraction",
      `${reportId}.probabilistic.uniqueOrigins`,
    );
    equal(repaired, 1, `${reportId} KQSP repaired validity`);
    summaries.push({
      id: reportId,
      examples: integer(unique.examples, `${reportId}.uniqueOrigins.examples`),
      candleMseSkillVsPersistence: requiredMetric(
        candle, "mseSkillVsPersistence", `${reportId}.uniqueOrigins.candle`,
      ),
      candleCorrelation: requiredMetric(
        candle, "anchoredLogCorrelation", `${reportId}.uniqueOrigins.candle`,
      ),
      closeReturnCorrelation: requiredMetric(
        closeReturn, "correlation", `${reportId}.uniqueOrigins.closeReturn`,
      ),
      horizonReturnIc: requiredMetric(
        paper, "horizonReturnIc", `${reportId}.uniqueOrigins.paperAligned`,
      ),
      priceSeriesIc: requiredMetric(
        paper, "priceSeriesIc", `${reportId}.uniqueOrigins.paperAligned`,
      ),
      oracleForwardKl: requiredMetric(
        oracle, "forwardKl", `${reportId}.uniqueOrigins.oracle`,
      ),
      samplePathCrps: requiredMetric(
        probabilisticUnique,
        "samplePathCrpsAnchoredLog",
        `${reportId}.probabilistic.uniqueOrigins`,
      ),
      repairedQuantileValidOhlcFraction: repaired,
    });
  }
  return { runSignature, models: summaries };
}

function auditInstalledModels(manifestSource: unknown): Array<{
  repoId: string;
  revision: string;
  bytes: number;
  configSha256: string;
  weightsSha256: string;
}> {
  const manifest = record(manifestSource, "model manifest");
  equal(manifest.version, 2, "model manifest.version");
  equal(record(manifest.source, "model manifest.source").commit, SOURCE_COMMIT, "model source commit");
  const snapshots = array(manifest.snapshots, "model manifest.snapshots");
  equal(snapshots.length, REQUIRED_SNAPSHOTS.size, "model manifest snapshot count");
  const installed: Array<{
    repoId: string;
    revision: string;
    bytes: number;
    configSha256: string;
    weightsSha256: string;
  }> = [];
  const seen = new Set<string>();
  for (const [index, value] of snapshots.entries()) {
    const snapshot = record(value, `model manifest.snapshots[${index}]`);
    const repoId = text(snapshot.repoId, `snapshots[${index}].repoId`);
    const revision = text(snapshot.revision, `snapshots[${index}].revision`);
    const expectedRevision = REQUIRED_SNAPSHOTS.get(repoId);
    if (!expectedRevision || revision !== expectedRevision || seen.has(repoId)) {
      throw new Error(`Unexpected, duplicate, or unpinned Kronos snapshot ${repoId}@${revision}.`);
    }
    seen.add(repoId);
    const relativePath = text(snapshot.path, `snapshots[${index}].path`);
    const directory = path.resolve(REPO_ROOT, relativePath);
    const weights = path.join(directory, "model.safetensors");
    const config = path.join(directory, "config.json");
    if (!fs.existsSync(weights) || !fs.existsSync(config)) {
      throw new Error(`Installed snapshot is incomplete: ${directory}`);
    }
    const actualBytes = fs.statSync(weights).size + fs.statSync(config).size;
    equal(actualBytes, snapshot.bytes, `${repoId} installed bytes`);
    const expectedHashes = SNAPSHOT_HASHES.get(repoId);
    if (!expectedHashes) throw new Error(`Missing expected hashes for ${repoId}.`);
    const configSha256 = sha256File(config);
    const weightsSha256 = sha256File(weights);
    equal(configSha256, expectedHashes.config, `${repoId} config SHA-256`);
    equal(weightsSha256, expectedHashes.weights, `${repoId} weights SHA-256`);
    const files = record(snapshot.files, `${repoId}.files`);
    const configIdentity = record(files["config.json"], `${repoId}.files.config.json`);
    const weightsIdentity = record(
      files["model.safetensors"],
      `${repoId}.files.model.safetensors`,
    );
    equal(configIdentity.bytes, fs.statSync(config).size, `${repoId} manifest config bytes`);
    equal(configIdentity.sha256, configSha256, `${repoId} manifest config SHA-256`);
    equal(weightsIdentity.bytes, fs.statSync(weights).size, `${repoId} manifest weights bytes`);
    equal(weightsIdentity.sha256, weightsSha256, `${repoId} manifest weights SHA-256`);
    installed.push({
      repoId,
      revision,
      bytes: actualBytes,
      configSha256,
      weightsSha256,
    });
  }
  equalSet([...seen], [...REQUIRED_SNAPSHOTS.keys()], "installed Kronos snapshots");
  return installed;
}

function auditFineTune(manifestSource: unknown, planSource: unknown, checkpointFile: string): {
  checkpointSha256: string;
  baselineValidationLoss: number;
  bestValidationLoss: number;
  baselineForecastLoss: number;
  bestForecastLoss: number;
  excludedRanges: number;
} {
  const manifest = record(manifestSource, "fine-tune manifest");
  equal(manifest.contract, "kronos-btcusdt-1m-predictor-finetune-v2", "fine-tune contract");
  equal(manifest.model, "base", "fine-tune model");
  equal(manifest.modelRevision, MODEL_SPECS[2].revision, "fine-tune model revision");
  equal(manifest.tokenizerRevision, MODEL_SPECS[2].tokenizerRevision, "fine-tune tokenizer revision");
  equal(manifest.tokenizerCheckpoint, null, "fine-tune frozen tokenizer");
  equal(manifest.lookback, 512, "fine-tune lookback");
  equal(manifest.horizon, 15, "fine-tune horizon");
  equal(manifest.exclusionPlanFingerprint, EXCLUSION_PLAN_SHA256, "fine-tune exclusion fingerprint");
  equal(manifest.trainCandidateStartsExcluded, 120_997, "excluded training starts");
  equal(manifest.validationCandidateStartsExcluded, 9_694, "excluded validation starts");
  const baselineValidationLoss = finite(manifest.baselineValidationLoss, "baselineValidationLoss");
  const bestValidationLoss = finite(manifest.bestValidationLoss, "bestValidationLoss");
  const baselineForecastLoss = finite(
    manifest.baselineValidationForecastLoss,
    "baselineValidationForecastLoss",
  );
  const bestForecastLoss = finite(manifest.bestValidationForecastLoss, "bestValidationForecastLoss");
  if (!(bestValidationLoss < baselineValidationLoss) || !(bestForecastLoss < baselineForecastLoss)) {
    throw new Error("Clean fine-tuning did not improve both validation objectives.");
  }
  const plan = record(planSource, "fine-tune exclusion plan");
  equal(plan.version, 1, "exclusion plan.version");
  equal(
    plan.contract,
    "kronos-btcusdt-1m-finetune-exclusion-plan-v1",
    "exclusion plan.contract",
  );
  equal(plan.validationCutoff, "2024-07-01T00:00:00.000Z", "exclusion validation cutoff");
  const ranges = array(plan.ranges, "exclusion plan.ranges");
  equal(ranges.length, 13, "exclusion range count");
  let previousEnd = Number.NEGATIVE_INFINITY;
  for (const [index, value] of ranges.entries()) {
    const range = record(value, `exclusion ranges[${index}]`);
    const start = Date.parse(text(range.start, `exclusion ranges[${index}].start`));
    const end = Date.parse(text(range.end, `exclusion ranges[${index}].end`));
    if (!Number.isFinite(start) || !Number.isFinite(end) || !(start < end) || start < previousEnd) {
      throw new Error(`Exclusion range ${index} is invalid or overlaps its predecessor.`);
    }
    previousEnd = end;
  }
  const checkpointSha256 = sha256File(checkpointFile);
  equal(checkpointSha256, CLEAN_CHECKPOINT_SHA256, "clean checkpoint SHA-256");
  return {
    checkpointSha256,
    baselineValidationLoss,
    bestValidationLoss,
    baselineForecastLoss,
    bestForecastLoss,
    excludedRanges: ranges.length,
  };
}

function auditExactValidationSplit(validation: ValidationIdentity): void {
  const windows = kronosInspectorWindows();
  const validationWindows = windows.filter((window) => window.startTime >= VALIDATION_CUTOFF);
  const expectedEpisodes = mergeOverlappingWindows(validationWindows, "validation");
  equal(validationWindows.length, 8, "expected validation inspector windows");
  equal(expectedEpisodes.length, 6, "expected merged validation episodes");
  equalSet(validation.split.windowIds, validationWindows.map((window) => window.id), "validation window ids");
  const byId = new Map(validation.episodes.map((episode) => [episode.windowId, episode]));
  equal(byId.size, expectedEpisodes.length, "unique validation episodes");
  for (const expected of expectedEpisodes) {
    const episode = byId.get(expected.id);
    if (!episode) throw new Error(`Validation is missing ${expected.id}.`);
    equal(episode.startTime, expected.startTime, `${expected.id}.startTime`);
    equal(episode.endTime, expected.endTime, `${expected.id}.endTime`);
  }
}

function auditEpisodeIdentities(
  source: unknown,
  expected: readonly { id: string; startTime: number; endTime: number; sourceWindowIds: string[] }[],
  label: string,
): void {
  const episodes = array(source, label);
  equal(episodes.length, expected.length, `${label}.length`);
  for (const [index, expectedEpisode] of expected.entries()) {
    const episode = record(episodes[index], `${label}[${index}]`);
    equal(episode.id, expectedEpisode.id, `${label}[${index}].id`);
    equal(episode.startTime, expectedEpisode.startTime, `${label}[${index}].startTime`);
    equal(episode.endTime, expectedEpisode.endTime, `${label}[${index}].endTime`);
    equalSet(
      array(episode.sourceWindowIds, `${label}[${index}].sourceWindowIds`).map(
        (value, windowIndex) => text(
          value,
          `${label}[${index}].sourceWindowIds[${windowIndex}]`,
        ),
      ),
      expectedEpisode.sourceWindowIds,
      `${label}[${index}].sourceWindowIds`,
    );
  }
}

export interface KronosAuditOptions {
  allModelsFile: string;
  modelManifestFile: string;
  fineTuneManifestFile: string;
  exclusionPlanFile: string;
  checkpointFile: string;
  forecastsFile: string;
  metricsFile: string;
  policyFile: string;
  validationFile: string;
}

export function auditKronosFinal(options: KronosAuditOptions): JsonRecord {
  const windows = kronosInspectorWindows();
  const windowIds = windows.map((window) => window.id);
  const modelInstallations = auditInstalledModels(loadJson(options.modelManifestFile));
  const allModels = auditBenchmarkReport(loadJson(options.allModelsFile), {
    expectedModels: MODEL_SPECS,
    expectedWindowIds: windowIds,
    expectedUniqueOrigins: 106,
    expectedMemberships: 112,
    expectedStride: 15,
  });
  const planSha256 = sha256File(options.exclusionPlanFile);
  equal(planSha256, EXCLUSION_PLAN_SHA256, "exclusion plan SHA-256");
  const fineTune = auditFineTune(
    loadJson(options.fineTuneManifestFile),
    loadJson(options.exclusionPlanFile),
    options.checkpointFile,
  );

  if (!fs.existsSync(options.forecastsFile)) {
    throw new Error(`Missing required artifact: ${options.forecastsFile}`);
  }
  const forecastBytes = fs.readFileSync(options.forecastsFile);
  const forecastSha256 = crypto.createHash("sha256").update(forecastBytes).digest("hex");
  const forecast = parseKronosForecastArtifact(JSON.parse(forecastBytes.toString("utf8")) as unknown);
  equal(forecast.modelId, "base-policy-holdout-pretrained-ensemble", "dense forecast model id");
  equal(forecast.rows.length, 11_232, "dense forecast row count");
  equal(forecast.lookbackCandles, 512, "dense forecast lookback");
  equal(forecast.horizonCandles, 15, "dense forecast horizon");
  equal(forecast.temperature, 0.8, "dense forecast temperature");
  equal(forecast.topP, 0.9, "dense forecast top-p");
  equal(forecast.sampleCount, 20, "dense forecast sample count");
  equal(forecast.originStrideCandles, 1, "dense forecast stride");
  validateDenseForecastCoverage(forecast, windows);
  const denseMetrics = auditBenchmarkReport(loadJson(options.metricsFile), {
    expectedModels: [MODEL_SPECS[2]],
    expectedWindowIds: windowIds,
    expectedUniqueOrigins: 11_232,
    expectedMemberships: 13_056,
    expectedStride: 1,
    expectedModelIds: ["base-policy-holdout-pretrained-ensemble"],
  });
  equal(denseMetrics.runSignature, forecast.runSignature, "dense metrics run signature");

  const policySource = record(loadJson(options.policyFile), "policy artifact");
  const policyArtifactSha256 = sha256File(options.policyFile);
  const policy = policySource as unknown as PolicyIdentity;
  assertPolicyGate(policy, forecast.runSignature);
  equal(policySource.version, 3, "policy.version");
  equal(
    policySource.selectionContract,
    "pretraining-ranking-with-untouched-post-training-confirmation-v2",
    "policy selection contract",
  );
  equal(policySource.forecastArtifactSha256, forecastSha256, "policy forecast SHA-256");
  equal(policySource.forecastModelId, forecast.modelId, "policy forecast model id");
  equal(policySource.validationCutoff, VALIDATION_CUTOFF, "policy validation cutoff");
  equal(policySource.candidateCount, 990, "policy candidate count");
  const selectedPolicy = record(policySource.selectedPolicy, "policy.selectedPolicy");
  equal(
    policy.selectedPolicyId,
    policyId(selectedPolicy as unknown as KronosBotPolicy),
    "selected policy id",
  );
  const calibrationWindowIds = array(
    policySource.calibrationWindowIds,
    "policy.calibrationWindowIds",
  ).map((value, index) => text(value, `policy.calibrationWindowIds[${index}]`));
  const expectedCalibrationWindows = windows.filter((window) =>
    window.startTime < VALIDATION_CUTOFF);
  equalSet(
    calibrationWindowIds,
    expectedCalibrationWindows.map((window) => window.id),
    "policy calibration window ids",
  );
  const expectedCalibrationEpisodes = mergeOverlappingWindows(
    expectedCalibrationWindows,
    "calibration",
  );
  const expectedPolicySplit = policyCalibrationEpisodeSplit(expectedCalibrationEpisodes);
  auditEpisodeIdentities(
    policySource.selectionEpisodes,
    expectedPolicySplit.selectionEpisodes,
    "policy.selectionEpisodes",
  );
  auditEpisodeIdentities(
    policySource.postTrainingConfirmationEpisodes,
    expectedPolicySplit.postTrainingConfirmationEpisodes,
    "policy.postTrainingConfirmationEpisodes",
  );
  equal(record(policySource.selection, "policy.selection").windows, 11, "selection aggregate windows");
  equal(
    record(policySource.postTrainingConfirmation, "policy.postTrainingConfirmation").windows,
    2,
    "confirmation aggregate windows",
  );
  const signalSource = text(selectedPolicy.signalSource, "policy.selectedPolicy.signalSource");
  if (signalSource === "calibrated-return") {
    const returnCalibration = record(
      policySource.returnCalibration,
      "policy.returnCalibration",
    );
    equal(
      returnCalibration.id,
      selectedPolicy.returnCalibrationId,
      "selected return-calibration id",
    );
    equalSet(
      array(
        returnCalibration.trainingEpisodeIds,
        "policy.returnCalibration.trainingEpisodeIds",
      ).map((value, index) => text(
        value,
        `policy.returnCalibration.trainingEpisodeIds[${index}]`,
      )),
      expectedPolicySplit.selectionEpisodes.map((episode) => episode.id),
      "return-calibration training episodes",
    );
  } else {
    equal(policySource.returnCalibration, null, "non-calibrated policy return calibration");
  }

  const validation = loadJson(options.validationFile) as ValidationIdentity;
  assertValidationReport(
    validation,
    forecast.runSignature,
    policy.selectedPolicyId,
    forecastSha256,
    policyArtifactSha256,
  );
  auditExactValidationSplit(validation);

  return {
    version: 1,
    contract: "kronos-final-completion-audit-v1",
    auditedAt: new Date().toISOString(),
    sourceCommit: SOURCE_COMMIT,
    files: {
      allModels: { path: options.allModelsFile, sha256: sha256File(options.allModelsFile) },
      exclusionPlan: { path: options.exclusionPlanFile, sha256: planSha256 },
      cleanCheckpoint: { path: options.checkpointFile, sha256: fineTune.checkpointSha256 },
      denseForecasts: { path: options.forecastsFile, sha256: forecastSha256 },
      denseMetrics: { path: options.metricsFile, sha256: sha256File(options.metricsFile) },
      policy: { path: options.policyFile, sha256: policyArtifactSha256 },
      validation: { path: options.validationFile, sha256: sha256File(options.validationFile) },
    },
    installation: { snapshots: modelInstallations },
    allModels: {
      runSignature: allModels.runSignature,
      windows: 28,
      uniqueOrigins: 106,
      models: allModels.models,
    },
    fineTune,
    dense: {
      runSignature: forecast.runSignature,
      rows: forecast.rows.length,
      windows: 28,
      windowMemberships: 13_056,
      metrics: denseMetrics.models[0],
    },
    policy: {
      id: policy.selectedPolicyId,
      candidateCount: 990,
      selection: policy.selection,
      postTrainingConfirmation: policy.postTrainingConfirmation,
    },
    validation: {
      episodes: validation.episodes.length,
      inspectorWindows: validation.split.windowIds.length,
      aggregate: validation.aggregate,
      gates: validation.gates,
      profitableHeldOut: validation.aggregate.netPnl > 0,
    },
    completion: {
      allPublicModelsInstalled: true,
      allEligibleInspectorWindowsMeasured: true,
      leakageFreeFineTuneVerified: true,
      denseCausalCoverageVerified: true,
      frozenPolicyVerified: true,
      heldOutBotFillsVerified: true,
      heldOutCostsAndRiskVerified: true,
    },
  };
}

function main(): void {
  const options: KronosAuditOptions = {
    allModelsFile: resolveArgument(
      "all-models",
      "data/benchmarks/kronos-calibration-all-t08-n20-n4.json",
    ),
    modelManifestFile: resolveArgument("model-manifest", ".tools/Kronos-models/manifest.json"),
    fineTuneManifestFile: resolveArgument(
      "finetune-manifest",
      ".tools/Kronos-finetuned/btcusdt-1m-base-policy-holdout-v2/manifest.json",
    ),
    exclusionPlanFile: resolveArgument(
      "exclusion-plan",
      "ml/training-plans/kronos-btcusdt-1m-policy-holdout-v1.json",
    ),
    checkpointFile: resolveArgument(
      "checkpoint",
      ".tools/Kronos-finetuned/btcusdt-1m-base-policy-holdout-v2/best_model/model.safetensors",
    ),
    forecastsFile: resolveArgument(
      "forecasts",
      "data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-forecasts.json",
    ),
    metricsFile: resolveArgument(
      "metrics",
      "data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-metrics.json",
    ),
    policyFile: resolveArgument("policy", "data/benchmarks/kronos-bot-policy-missing.json"),
    validationFile: resolveArgument(
      "validation",
      "data/benchmarks/kronos-bot-validation-missing.json",
    ),
  };
  const audit = auditKronosFinal(options);
  const signature = text(record(audit.dense, "audit.dense").runSignature, "audit run signature");
  const output = resolveArgument(
    "output",
    `data/benchmarks/kronos-final-audit-${signature.slice(0, 12)}.json`,
  );
  atomicJson(output, audit);
  console.log(JSON.stringify({ output, ...audit }, null, 2));
}

if (path.resolve(process.argv[1] ?? "") === import.meta.filename) {
  try {
    main();
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}
