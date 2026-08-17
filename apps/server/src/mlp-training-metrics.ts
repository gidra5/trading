import fs from "node:fs/promises";
import path from "node:path";

const MAX_LOG_CHUNK_BYTES = 128 * 1024 * 1024;
const METRIC_EVENTS = new Set([
  "dataset-complete",
  "dataset-component",
  "dataset-example-weighting",
  "dataset-feature-refresh-complete",
  "dataset-feature-refresh-progress",
  "dataset-oracle",
  "dataset-progress",
  "dataset-shard",
  "dataset-source-recovered",
  "dataset-source-recovery-complete",
  "dataset-source-recovery-start",
  "dataset-source-rejected",
  "dataset-stage-timing",
  "epoch",
  "baseline",
  "train-step",
  "training-complete",
  "training-start",
  "time-weighting-ready",
]);
const METRIC_EVENT_ALIASES = new Map([
  ["minute-return-dataset-selected", "dataset-complete"],
  ["minute-return-epoch", "epoch"],
  ["minute-return-complete", "training-complete"],
]);

interface TrainingPlan {
  id: string;
  label: string;
  runDir: string;
  datasetDir?: string;
  dataset?: {
    datasetDir?: string;
  };
  architecture?: {
    dropout?: number;
    dropoutRate?: number;
  };
  samplingIntervalMs?: number;
  predictionDelayMs?: number;
  archived?: boolean;
  archivedAt?: string;
  bestValidationKl?: number;
  bestValidationKlEpoch?: number;
  training?: {
    epochs?: number;
    patience?: number;
    earlyStoppingPatience?: number;
    dropout?: number;
    dropoutRate?: number;
    lossWeights?: Record<string, number>;
    reverseKl?: MlpReverseKlPlan;
    outputRegularizer?: MlpOutputRegularizerPlan;
    softWeightBound?: MlpSoftWeightBoundPlan;
    branchNormalization?: MlpBranchNormalizationPlan;
    timeWeighting?: MlpTimeWeightingPlan;
  };
}

interface TrainingRunCatalog {
  runs: TrainingPlan[];
}

interface TrainingRunDisplayMetadata {
  label?: string;
}

interface TrainingMatrixManifest {
  id: string;
  label: string;
  runIds: string[];
  control?: {
    pauseFile: string;
  };
  artifactProgress?: {
    completionFile: string;
    statusFile: string;
    completionContract?: string;
  };
}

export interface MlpReverseKlPlan {
  predictionMixtureWeight: number;
}

export interface MlpOutputRegularizerPlan {
  applicationProbabilities: {
    reverseKl: number;
    entropySharpness: number;
  };
  samplingUnit: "optimizer-update";
  independentGates: boolean;
  inverseProbabilityScaling: boolean;
}

export interface MlpSoftWeightBoundPlan {
  desiredMagnitude: number;
  sharpness: number;
  absoluteEpsilon: number;
}

export interface MlpBranchNormalizationPlan {
  learnableCentering?: boolean;
}

export interface MlpTimeWeightingPlan {
  mode: "distanceImbalance";
  distanceEpsilon: number;
  minimumWeight: number;
  stateAggregation: "globalDistanceRatio";
  minimumAdviceMagnitude: number;
  memoryHalfLifeSteps: number;
  growthPerPriorAdvice: number;
  maximumMultiplier: number;
  resetAfterGapSteps: number;
  resolutionDivergenceMultiplier: number;
}

interface TrainingStatus {
  pid?: number;
  stage?: string;
  startedAt?: string;
  updatedAt?: string;
  completedAt?: string;
  failedAt?: string;
  pausedAt?: string;
  message?: string;
  error?: string;
  latest?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface MlpTrainingMetricEvent {
  event: string;
  [key: string]: unknown;
}

export interface MlpTrainingRunSummary {
  key: string;
  id: string;
  label: string;
  running: boolean;
  stage?: string;
  updatedAt?: string;
  epochs?: number;
  patience?: number;
  archived?: boolean;
  bestValidationKl?: number;
  bestValidationKlEpoch?: number;
}

export interface MlpTrainingMatrixProgress {
  id: string;
  label: string;
  totalRuns: number;
  completedRuns: number;
  failedRuns: number;
  queuedRuns: number;
  progress: number;
  controllable: boolean;
  pauseRequested: boolean;
  paused: boolean;
  active?: {
    id: string;
    label: string;
    stage: string;
    epoch?: number;
    epochs?: number;
    bestTrainScore?: number;
  };
}

export interface MlpTrainingMetricsResponse {
  runs: MlpTrainingRunSummary[];
  matrices: MlpTrainingMatrixProgress[];
  selectedRunKey: string;
  plan: {
    id: string;
    label: string;
    epochs?: number;
    patience?: number;
    samplingIntervalMs?: number;
    predictionDelayMs?: number;
    archived?: boolean;
    archivedAt?: string;
    bestValidationKl?: number;
    bestValidationKlEpoch?: number;
    dropout?: number;
    dropoutRate?: number;
    lossWeights?: Record<string, number>;
    reverseKl?: MlpReverseKlPlan;
    outputRegularizer?: MlpOutputRegularizerPlan;
    softWeightBound?: MlpSoftWeightBoundPlan;
    branchNormalization?: MlpBranchNormalizationPlan;
    timeWeighting?: MlpTimeWeightingPlan;
  };
  status?: TrainingStatus;
  running: boolean;
  finalizeRequested: boolean;
  progress: {
    refinedShards?: number;
    totalShards?: number;
    remainingTeacherFits?: number;
    sourceRejectedDays?: number;
    featureComponents?: number;
    oracleComponents?: number;
  };
  cursor: number;
  reset: boolean;
  events: MlpTrainingMetricEvent[];
}

export interface MlpTrainingComparisonMetricValues {
  normalizedMse?: number;
  mse?: number;
  rmse?: number;
  mae?: number;
  zeroBaselineMse?: number;
  mseSkillVsZero?: number;
  directionAccuracy?: number;
  correlation?: number;
}

export interface MlpTrainingComparisonDistributionValues {
  negativeLogLikelihood?: number;
  unitNegativeLogLikelihood?: number;
  bitsPerExample?: number;
  globalBaselineNegativeLogLikelihood?: number;
  nllImprovementVsGlobal?: number;
  rawNegativeLogLikelihood?: number;
  nllImprovementVsRaw?: number;
  expectation?: MlpTrainingComparisonMetricValues;
  perLeadExpectation?: MlpTrainingComparisonMetricValues[];
  mode?: MlpTrainingComparisonMetricValues;
}

export interface MlpTrainingOutputCalibrationValues {
  scale?: number;
  affineScale?: number;
  affineIntercept?: number;
  calibrationCorrelation?: number;
  validation?: MlpTrainingComparisonMetricValues;
  test?: MlpTrainingComparisonMetricValues;
  affineValidation?: MlpTrainingComparisonMetricValues;
  affineTest?: MlpTrainingComparisonMetricValues;
  densityTemperature?: number;
  densityValidation?: MlpTrainingComparisonDistributionValues;
  densityTest?: MlpTrainingComparisonDistributionValues;
}

export interface MlpTrainingAutoregressiveEpisodeValues {
  episodes?: number;
  sourceEpisodeSeconds?: number;
  activeCandles?: number;
  activeCandlesPerEpisode?: {
    minimum?: number;
    mean?: number;
    maximum?: number;
  };
  pooledCandles?: MlpTrainingComparisonMetricValues;
  episodeAverage?: MlpTrainingComparisonMetricValues;
  episodeReturnCorrelation?: number;
  episodeCumulativePathCorrelation?: number;
  episodeEndpoint?: MlpTrainingComparisonMetricValues;
  estimator?: {
    trajectories?: number;
    randomizedReplicates?: number;
    trajectoriesPerReplicate?: number;
    returnTrajectoryVarianceMean?: number;
    returnMean?: MlpTrainingEstimatorVarianceValues;
    cumulativeLogPricePathMean?: MlpTrainingEstimatorVarianceValues;
    endpointTrajectoryVarianceMean?: number;
    endpointMean?: MlpTrainingEstimatorVarianceValues;
  };
  pathLikelihood?: {
    exactPathProbabilityMass?: number;
    episodes?: number;
    realizedNegativeLogDensityPerCandle?: MlpTrainingLikelihoodSummaryValues;
    realizedBitsPerCandle?: MlpTrainingLikelihoodSummaryValues;
    sampledPathLogDensityPercentile?: MlpTrainingLikelihoodSummaryValues;
    twoSidedTypicality?: MlpTrainingLikelihoodSummaryValues;
    logDensityZScoreVsSampledPaths?: MlpTrainingLikelihoodSummaryValues;
    perCandleDensityRatioVsSampleMedian?: MlpTrainingLikelihoodSummaryValues;
  };
}

export interface MlpTrainingEstimatorVarianceValues {
  meanVariance?: number;
  meanStandardError?: number;
  p95StandardError?: number;
  maximumStandardError?: number;
}

export interface MlpTrainingLikelihoodSummaryValues {
  mean?: number;
  median?: number;
  p95?: number;
  minimum?: number;
  fractionBelow1Percent?: number;
  fractionBelow5Percent?: number;
}

export interface MlpTrainingCheckpointSelectionValues {
  epoch: number;
  selectionScore?: number;
  train?: MlpTrainingComparisonMetricValues;
  validation?: MlpTrainingComparisonMetricValues;
  test?: MlpTrainingComparisonMetricValues;
  distribution?: {
    train?: MlpTrainingComparisonDistributionValues;
    validation?: MlpTrainingComparisonDistributionValues;
    test?: MlpTrainingComparisonDistributionValues;
  };
  autoregressiveEpisodes?: {
    validation?: MlpTrainingAutoregressiveEpisodeValues;
    test?: MlpTrainingAutoregressiveEpisodeValues;
  };
  sobolExpectedEpisodes?: {
    validation?: MlpTrainingAutoregressiveEpisodeValues;
    test?: MlpTrainingAutoregressiveEpisodeValues;
  };
  outputCalibration?: MlpTrainingOutputCalibrationValues;
}

export interface MlpTrainingComparisonRun {
  key: string;
  id: string;
  label: string;
  running: boolean;
  stage?: string;
  epoch?: number;
  epochs?: number;
  examples?: number;
  parameterCount?: number;
  trainableParameterCount?: number;
  bestEpoch?: number;
  train?: MlpTrainingComparisonMetricValues;
  validation?: MlpTrainingComparisonMetricValues;
  test?: MlpTrainingComparisonMetricValues;
  distribution?: {
    train?: MlpTrainingComparisonDistributionValues;
    validation?: MlpTrainingComparisonDistributionValues;
    test?: MlpTrainingComparisonDistributionValues;
  };
  checkpointSelections?: Record<string, MlpTrainingCheckpointSelectionValues>;
  outputCalibration?: MlpTrainingOutputCalibrationValues;
  validationCheckpointEpoch?: number;
  fit: Array<{
    epoch: number;
    trainNormalizedMse?: number;
    validationNormalizedMse?: number;
    trainNegativeLogLikelihood?: number;
    validationNegativeLogLikelihood?: number;
    bestTrainScore?: number;
  }>;
}

export interface MlpTrainingComparisonResponse {
  runs: MlpTrainingComparisonRun[];
}

export class MlpTrainingRunNotFoundError extends Error {
  constructor(readonly runKey: string) {
    super(`Unknown MLP training run: ${runKey}`);
    this.name = "MlpTrainingRunNotFoundError";
  }
}

interface LoadedTrainingPlan {
  key: string;
  plan: TrainingPlan;
  runDir: string;
  datasetDir: string;
  statusFile: string;
  logFiles: string[];
  finalizeFile: string;
  status?: TrainingStatus;
  updatedAt?: string;
  running: boolean;
}

/** Incrementally exposes the append-only MLP run log to the local dashboard. */
export class MlpTrainingMetricsReader {
  private readonly repoRoot: string;

  constructor(
    private readonly planFile: string,
    repoRoot = path.resolve(path.dirname(planFile), ".."),
  ) {
    this.repoRoot = path.resolve(repoRoot);
  }

  async read(
    cursor: number,
    requestedRunKey?: string,
  ): Promise<MlpTrainingMetricsResponse> {
    const availableRuns = await this.discoverRuns();
    const files = requestedRunKey
      ? availableRuns.find((candidate) => candidate.key === requestedRunKey)
      : availableRuns[0];
    if (!files) {
      throw new MlpTrainingRunNotFoundError(requestedRunKey ?? "");
    }
    const [log, progress, queue, sourceQueue, finalizeRequested] = await Promise.all([
      readMetricLogs(files.logFiles, cursor),
      readOptionalJson<{
        shards?: Array<{ refinementPass?: number }>;
        featureComponents?: unknown[];
        oracleComponents?: unknown[];
      }>(
        path.join(files.datasetDir, "state", "progress.json"),
      ),
      readOptionalJson<{ cases?: unknown[] }>(
        path.join(files.datasetDir, "state", "teacher-refinement-queue.json"),
      ),
      readOptionalJson<{ cases?: unknown[] }>(
        path.join(files.datasetDir, "state", "source-rejection-queue.json"),
      ),
      exists(files.finalizeFile),
    ]);
    const shards = Array.isArray(progress?.shards) ? progress.shards : undefined;
    return {
      runs: availableRuns.map((candidate) => ({
        key: candidate.key,
        id: candidate.plan.id,
        label: candidate.plan.label,
        running: candidate.running,
        ...(candidate.status?.stage ? { stage: candidate.status.stage } : {}),
        ...(candidate.updatedAt ? { updatedAt: candidate.updatedAt } : {}),
        ...(candidate.plan.training?.epochs === undefined
          ? {}
          : { epochs: candidate.plan.training.epochs }),
        ...((candidate.plan.training?.patience
          ?? candidate.plan.training?.earlyStoppingPatience) === undefined
          ? {}
          : {
              patience: candidate.plan.training?.patience
                ?? candidate.plan.training?.earlyStoppingPatience,
            }),
        ...(candidate.plan.archived ? { archived: true } : {}),
        ...(candidate.plan.bestValidationKl === undefined
          ? {}
          : { bestValidationKl: candidate.plan.bestValidationKl }),
        ...(candidate.plan.bestValidationKlEpoch === undefined
          ? {}
          : { bestValidationKlEpoch: candidate.plan.bestValidationKlEpoch }),
      })),
      matrices: await this.discoverMatrices(availableRuns),
      selectedRunKey: files.key,
      plan: {
        id: files.plan.id,
        label: files.plan.label,
        ...(files.plan.training?.epochs === undefined
          ? {}
          : { epochs: files.plan.training.epochs }),
        ...((files.plan.training?.patience
          ?? files.plan.training?.earlyStoppingPatience) === undefined
          ? {}
          : {
              patience: files.plan.training?.patience
                ?? files.plan.training?.earlyStoppingPatience,
            }),
        ...(files.plan.samplingIntervalMs === undefined
          ? {}
          : { samplingIntervalMs: files.plan.samplingIntervalMs }),
        ...((typeof files.status?.predictionDelayMs === "number"
          ? files.status.predictionDelayMs
          : files.plan.predictionDelayMs) === undefined
          ? {}
          : {
              predictionDelayMs: typeof files.status?.predictionDelayMs === "number"
                ? files.status.predictionDelayMs
                : files.plan.predictionDelayMs,
            }),
        ...(files.plan.archived ? { archived: true } : {}),
        ...(files.plan.archivedAt ? { archivedAt: files.plan.archivedAt } : {}),
        ...(files.plan.bestValidationKl === undefined
          ? {}
          : { bestValidationKl: files.plan.bestValidationKl }),
        ...(files.plan.bestValidationKlEpoch === undefined
          ? {}
          : { bestValidationKlEpoch: files.plan.bestValidationKlEpoch }),
        ...((files.plan.training?.dropout
          ?? files.plan.architecture?.dropout) === undefined
          ? {}
          : {
              dropout: files.plan.training?.dropout
                ?? files.plan.architecture?.dropout,
            }),
        ...((files.plan.training?.dropoutRate
          ?? files.plan.architecture?.dropoutRate) === undefined
          ? {}
          : {
              dropoutRate: files.plan.training?.dropoutRate
                ?? files.plan.architecture?.dropoutRate,
            }),
        ...(files.plan.training?.lossWeights
          ? { lossWeights: files.plan.training.lossWeights }
          : {}),
        ...(files.plan.training?.reverseKl
          ? { reverseKl: files.plan.training.reverseKl }
          : {}),
        ...(files.plan.training?.outputRegularizer
          ? { outputRegularizer: files.plan.training.outputRegularizer }
          : {}),
        ...(files.plan.training?.softWeightBound
          ? { softWeightBound: files.plan.training.softWeightBound }
          : {}),
        ...(files.plan.training?.branchNormalization
          ? { branchNormalization: files.plan.training.branchNormalization }
          : {}),
        ...(files.plan.training?.timeWeighting
          ? { timeWeighting: files.plan.training.timeWeighting }
          : {}),
      },
      ...(files.status ? { status: files.status } : {}),
      running: files.running,
      finalizeRequested,
      progress: {
        ...(shards ? {
          totalShards: shards.length,
          refinedShards: shards.filter((shard) => (shard.refinementPass ?? 0) > 0).length,
        } : {}),
        ...(Array.isArray(queue?.cases) ? { remainingTeacherFits: queue.cases.length } : {}),
        ...(Array.isArray(sourceQueue?.cases)
          ? { sourceRejectedDays: sourceQueue.cases.length }
          : {}),
        ...(Array.isArray(progress?.featureComponents)
          ? { featureComponents: progress.featureComponents.length }
          : {}),
        ...(Array.isArray(progress?.oracleComponents)
          ? { oracleComponents: progress.oracleComponents.length }
          : {}),
      },
      cursor: log.cursor,
      reset: log.reset,
      events: log.events,
    };
  }

  async compare(runKeys: readonly string[]): Promise<MlpTrainingComparisonResponse> {
    const availableRuns = await this.discoverRuns();
    const byKey = new Map(availableRuns.map((run) => [run.key, run]));
    const selected = runKeys.map((runKey) => {
      const run = byKey.get(runKey);
      if (!run) throw new MlpTrainingRunNotFoundError(runKey);
      return run;
    });
    return {
      runs: await Promise.all(selected.map(async (files) => {
        const [storedResult, rootResult, validation, outputCalibration,
          checkpointSelection, checkpointCalibrations, log] = await Promise.all([
          readOptionalJson<Record<string, unknown>>(
            path.join(files.runDir, "state", "result.json"),
          ),
          readOptionalJson<Record<string, unknown>>(
            path.join(files.runDir, "result.json"),
          ),
          readOptionalJson<Record<string, unknown>>(
            path.join(files.runDir, "state", "validation-current-best.json"),
          ),
          readOptionalJson<Record<string, unknown>>(
            path.join(
              files.runDir,
              "state",
              "output-calibration-pre-validation-7d.json",
            ),
          ),
          readOptionalJson<Record<string, unknown>>(
            path.join(
              files.runDir,
              "state",
              "checkpoint-selection-comparison.json",
            ),
          ),
          readOptionalJson<Record<string, unknown>>(
            path.join(
              files.runDir,
              "state",
              "checkpoint-selection-calibrations.json",
            ),
          ),
          readMetricLogs(files.logFiles, 0),
        ]);
        const result = storedResult ?? rootResult;
        const resultTrain = metricRecord(result?.train);
        const resultValidation = metricRecord(
          result?.bestValidation
            ?? result?.validation
            ?? result?.validationMetrics
            ?? result?.ridgeValidation,
        );
        const externalValidation = metricRecord(validation?.metrics);
        const selectedValidation = externalValidation ?? resultValidation;
        const parsedOutputCalibration = outputCalibrationRecord(outputCalibration);
        const resultTest = metricRecord(result?.test);
        const resultDistribution = recordField(result?.distribution);
        const trainDistribution = distributionRecord(resultDistribution?.train);
        const validationDistribution = distributionRecord(
          resultDistribution?.validation,
        );
        const testDistribution = distributionRecord(resultDistribution?.test);
        const checkpointSelections: Record<
          string, MlpTrainingCheckpointSelectionValues
        > = {};
        const checkpointPolicies = recordField(checkpointSelection?.policies);
        const checkpointCalibrationPolicies = recordField(
          checkpointCalibrations?.policies,
        );
        for (const [policy, value] of Object.entries(checkpointPolicies ?? {})) {
          const source = recordField(value);
          const epoch = numberField(source?.epoch);
          if (!source || epoch === undefined) continue;
          const policyDistribution = recordField(source.distribution);
          const trainPolicyDistribution = distributionRecord(policyDistribution?.train);
          const validationPolicyDistribution = distributionRecord(
            policyDistribution?.validation,
          );
          const testPolicyDistribution = distributionRecord(policyDistribution?.test);
          const policyOutputCalibration = outputCalibrationRecord(
            checkpointCalibrationPolicies?.[policy],
          );
          const autoregressiveEpisodes = recordField(
            source.autoregressiveEpisodes,
          );
          const validationEpisodes = autoregressiveEpisodeRecord(
            autoregressiveEpisodes?.validation,
          );
          const testEpisodes = autoregressiveEpisodeRecord(
            autoregressiveEpisodes?.test,
          );
          const sobolExpectedEpisodes = recordField(
            source.sobolExpectedEpisodes,
          );
          const validationSobolEpisodes = autoregressiveEpisodeRecord(
            sobolExpectedEpisodes?.validation,
          );
          const testSobolEpisodes = autoregressiveEpisodeRecord(
            sobolExpectedEpisodes?.test,
          );
          checkpointSelections[policy] = {
            epoch,
            ...(numberField(source.selectionScore) === undefined
              ? {}
              : { selectionScore: numberField(source.selectionScore) }),
            ...(metricRecord(source.train) ? { train: metricRecord(source.train) } : {}),
            ...(metricRecord(source.validation)
              ? { validation: metricRecord(source.validation) }
              : {}),
            ...(metricRecord(source.test) ? { test: metricRecord(source.test) } : {}),
            ...(trainPolicyDistribution
              || validationPolicyDistribution
              || testPolicyDistribution
              ? {
                  distribution: {
                    ...(trainPolicyDistribution
                      ? { train: trainPolicyDistribution }
                      : {}),
                    ...(validationPolicyDistribution
                      ? { validation: validationPolicyDistribution }
                      : {}),
                    ...(testPolicyDistribution
                      ? { test: testPolicyDistribution }
                      : {}),
                  },
                }
              : {}),
            ...(validationEpisodes || testEpisodes
              ? {
                  autoregressiveEpisodes: {
                    ...(validationEpisodes
                      ? { validation: validationEpisodes }
                      : {}),
                    ...(testEpisodes ? { test: testEpisodes } : {}),
                  },
                }
              : {}),
            ...(validationSobolEpisodes || testSobolEpisodes
              ? {
                  sobolExpectedEpisodes: {
                    ...(validationSobolEpisodes
                      ? { validation: validationSobolEpisodes }
                      : {}),
                    ...(testSobolEpisodes
                      ? { test: testSobolEpisodes }
                      : {}),
                  },
                }
              : {}),
            ...(policyOutputCalibration
              ? { outputCalibration: policyOutputCalibration }
              : {}),
          };
        }
        const examples = numberField(result?.examples);
        const parameterCount = numberField(result?.parameterCount);
        const trainableParameterCount = numberField(result?.trainableParameterCount);
        const bestEpoch = numberField(result?.bestEpoch);
        const currentEpoch = numberField(files.status?.latest?.epoch)
          ?? numberField(files.status?.epoch);
        const validationCheckpointEpoch = numberField(validation?.checkpointEpoch);
        const fit: MlpTrainingComparisonRun["fit"] = log.events.flatMap((event) => {
          if (event.event !== "epoch") return [];
          const epoch = numberField(event.epoch);
          if (epoch === undefined) return [];
          const train = metricRecord(event.train);
          const validationMetrics = metricRecord(event.validation);
          const trainDensity = distributionRecord(event.trainDistribution);
          const validationDensity = distributionRecord(event.validationDistribution);
          const bestTrainScore = numberField(event.bestTrainScore);
          return [{
            epoch,
            ...(train?.normalizedMse === undefined
              ? {}
              : { trainNormalizedMse: train.normalizedMse }),
            ...(validationMetrics?.normalizedMse === undefined
              ? {}
              : { validationNormalizedMse: validationMetrics.normalizedMse }),
            ...(trainDensity?.negativeLogLikelihood === undefined
              ? {}
              : { trainNegativeLogLikelihood: trainDensity.negativeLogLikelihood }),
            ...(validationDensity?.negativeLogLikelihood === undefined
              ? {}
              : {
                  validationNegativeLogLikelihood:
                    validationDensity.negativeLogLikelihood,
                }),
            ...(bestTrainScore === undefined ? {} : { bestTrainScore }),
          }];
        });
        if (selectedValidation?.normalizedMse !== undefined
          && !fit.some((point) => point.validationNormalizedMse !== undefined)) {
          const checkpointEpoch = validationCheckpointEpoch ?? bestEpoch;
          if (checkpointEpoch !== undefined) {
            const existing = fit.find((point) => point.epoch === checkpointEpoch);
            if (existing) {
              existing.validationNormalizedMse = selectedValidation.normalizedMse;
            } else {
              fit.push({
                epoch: checkpointEpoch,
                validationNormalizedMse: selectedValidation.normalizedMse,
              });
              fit.sort((left, right) => left.epoch - right.epoch);
            }
          }
        }
        return {
          key: files.key,
          id: files.plan.id,
          label: files.plan.label,
          running: files.running,
          ...(files.status?.stage ? { stage: files.status.stage } : {}),
          ...(currentEpoch === undefined ? {} : { epoch: currentEpoch }),
          ...(files.plan.training?.epochs === undefined
            ? {}
            : { epochs: files.plan.training.epochs }),
          ...(examples === undefined ? {} : { examples }),
          ...(parameterCount === undefined ? {} : { parameterCount }),
          ...(trainableParameterCount === undefined ? {} : { trainableParameterCount }),
          ...(bestEpoch === undefined ? {} : { bestEpoch }),
          ...(resultTrain ? { train: resultTrain } : {}),
          ...(selectedValidation
            ? { validation: selectedValidation }
            : {}),
          ...(resultTest ? { test: resultTest } : {}),
          ...(Object.keys(checkpointSelections).length > 0
            ? { checkpointSelections }
            : {}),
          ...(trainDistribution || validationDistribution || testDistribution
            ? {
                distribution: {
                  ...(trainDistribution ? { train: trainDistribution } : {}),
                  ...(validationDistribution
                    ? { validation: validationDistribution }
                    : {}),
                  ...(testDistribution ? { test: testDistribution } : {}),
                },
              }
            : {}),
          ...(parsedOutputCalibration
            ? { outputCalibration: parsedOutputCalibration }
            : {}),
          ...(validationCheckpointEpoch === undefined
            ? {}
            : { validationCheckpointEpoch }),
          fit,
        };
      })),
    };
  }

  async setMatrixPaused(
    matrixId: string,
    paused: boolean,
  ): Promise<{ matrixId: string; pauseRequested: boolean }> {
    const directory = path.join(this.repoRoot, "ml", "training-matrices");
    let entries: Array<{ name: string; isFile(): boolean }>;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        throw new MlpTrainingMatrixControlError(`Unknown training matrix: ${matrixId}`);
      }
      throw error;
    }
    let selected: TrainingMatrixManifest | undefined;
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
      try {
        const candidate = JSON.parse(
          await fs.readFile(path.join(directory, entry.name), "utf8"),
        ) as TrainingMatrixManifest;
        if (candidate.id === matrixId) {
          selected = candidate;
          break;
        }
      } catch {
        continue;
      }
    }
    if (!selected) {
      throw new MlpTrainingMatrixControlError(`Unknown training matrix: ${matrixId}`);
    }
    if (!selected.control?.pauseFile) {
      throw new MlpTrainingMatrixControlError(
        `Training matrix does not support pause/resume: ${matrixId}`,
      );
    }
    const controlRoot = path.resolve(
      this.repoRoot,
      "data",
      "training",
      "matrices",
    );
    const pauseFile = path.resolve(this.repoRoot, selected.control.pauseFile);
    const relative = path.relative(controlRoot, pauseFile);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new MlpTrainingMatrixControlError(
        `Training matrix has an invalid pause control path: ${matrixId}`,
      );
    }
    if (paused) {
      await fs.mkdir(path.dirname(pauseFile), { recursive: true });
      await fs.writeFile(pauseFile, `${JSON.stringify({
        matrixId,
        requestedAt: new Date().toISOString(),
      }, null, 2)}\n`, { flag: "w" });
    } else {
      await fs.rm(pauseFile, { force: true });
    }
    return { matrixId, pauseRequested: paused };
  }

  private async discoverRuns(): Promise<LoadedTrainingPlan[]> {
    const planFiles = new Set<string>([path.resolve(this.planFile)]);
    const planDirectory = path.join(this.repoRoot, "ml", "training-plans");
    try {
      for (const entry of await fs.readdir(planDirectory, { withFileTypes: true })) {
        if (entry.isFile() && entry.name.endsWith(".json")) {
          planFiles.add(path.join(planDirectory, entry.name));
        }
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const loaded = await Promise.all([...planFiles].map(async (planFile) => {
      try {
        return await this.loadPlan(planFile);
      } catch (error) {
        if (path.resolve(planFile) === path.resolve(this.planFile)) throw error;
        return undefined;
      }
    }));
    const configured = loaded.filter(
      (candidate): candidate is LoadedTrainingPlan => candidate !== undefined,
    );
    const configuredRunDirs = new Set(configured.map((candidate) => candidate.runDir));
    const snapshots = (await this.discoverRunSnapshots()).filter(
      (candidate) => !configuredRunDirs.has(candidate.runDir),
    );
    const catalog = await readOptionalJson<TrainingRunCatalog>(
      path.join(this.repoRoot, "ml", "training-run-catalog.json"),
    );
    const archived = await Promise.all(
      (catalog?.runs ?? []).map((plan) => this.loadArchivedRun(plan)),
    );
    return [
      ...configured,
      ...snapshots,
      ...archived,
    ].sort(compareRuns);
  }

  private async discoverRunSnapshots(): Promise<LoadedTrainingPlan[]> {
    const runsRoot = path.join(this.repoRoot, "data", "training", "runs");
    let entries: Array<{
      name: string;
      isDirectory(): boolean;
      isFile(): boolean;
    }>;
    try {
      entries = await fs.readdir(runsRoot, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
    const loaded = await Promise.all(entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const stateDir = path.join(runsRoot, entry.name, "state");
        const snapshotFile = path.join(stateDir, "plan.json");
        try {
          const [stored, display] = await Promise.all([
            fs.readFile(snapshotFile, "utf8").then((value) => JSON.parse(value) as {
              plan?: TrainingPlan;
            } & Partial<TrainingPlan>),
            readOptionalJson<TrainingRunDisplayMetadata>(
              path.join(stateDir, "display.json"),
            ),
          ]);
          const storedPlan = stored.plan ?? stored as TrainingPlan;
          const displayLabel = display?.label?.trim();
          const plan = displayLabel
            ? { ...storedPlan, label: displayLabel }
            : storedPlan;
          return await this.loadStoredPlan(plan, `run/${plan.id}`, snapshotFile);
        } catch {
          return undefined;
        }
      }));
    return loaded.filter(
      (candidate): candidate is LoadedTrainingPlan => candidate !== undefined,
    );
  }

  private async discoverMatrices(
    runs: readonly LoadedTrainingPlan[],
  ): Promise<MlpTrainingMatrixProgress[]> {
    const directory = path.join(this.repoRoot, "ml", "training-matrices");
    let entries: Array<{
      name: string;
      isDirectory(): boolean;
      isFile(): boolean;
    }>;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
    const byId = new Map(runs.map((run) => [run.plan.id, run]));
    const matrices: MlpTrainingMatrixProgress[] = [];
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
      try {
        const manifest = JSON.parse(
          await fs.readFile(path.join(directory, entry.name), "utf8"),
        ) as TrainingMatrixManifest;
        if (!manifest.id || !manifest.label || !Array.isArray(manifest.runIds)
          || manifest.runIds.length === 0) continue;
        const matrixRuns = manifest.runIds.map((id) => byId.get(id));
        const artifactStates = manifest.artifactProgress
          ? await Promise.all(manifest.runIds.map(async (id) => {
              const completionFile = path.resolve(
                this.repoRoot,
                manifest.artifactProgress!.completionFile.replaceAll("{runId}", id),
              );
              const statusFile = path.resolve(
                this.repoRoot,
                manifest.artifactProgress!.statusFile.replaceAll("{runId}", id),
              );
              const requiredContract = (
                manifest.artifactProgress!.completionContract
              );
              const completion = requiredContract
                ? await readOptionalJson<{ contract?: string }>(completionFile)
                : undefined;
              return {
                id,
                complete: requiredContract
                  ? completion?.contract === requiredContract
                  : await exists(completionFile),
                status: await readOptionalJson<TrainingStatus>(statusFile),
              };
            }))
          : undefined;
        const completedRuns = artifactStates
          ? artifactStates.filter((value) => value.complete).length
          : matrixRuns.filter((run) => run?.status?.stage === "complete").length;
        const failedRuns = artifactStates
          ? artifactStates.filter((value) => value.status?.stage === "failed").length
          : matrixRuns.filter((run) => run?.status?.stage === "failed").length;
        const pauseFile = manifest.control?.pauseFile
          ? path.resolve(this.repoRoot, manifest.control.pauseFile)
          : undefined;
        const controlRoot = path.resolve(
          this.repoRoot,
          "data",
          "training",
          "matrices",
        );
        const relativePauseFile = pauseFile
          ? path.relative(controlRoot, pauseFile)
          : undefined;
        const controllable = Boolean(
          pauseFile
          && relativePauseFile
          && !relativePauseFile.startsWith("..")
          && !path.isAbsolute(relativePauseFile),
        );
        const pauseRequested = controllable && pauseFile
          ? await exists(pauseFile)
          : false;
        const activeArtifact = artifactStates?.find((value) =>
          !value.complete
          && processIsAlive(value.status?.pid)
          && !isTerminalStage(value.status?.stage));
        const active = artifactStates
          ? undefined
          : matrixRuns.find((run) => run?.running);
        const pausedRun = !artifactStates && pauseRequested
          ? matrixRuns.find((run) => run?.status?.stage === "paused")
          : undefined;
        const current = active ?? pausedRun;
        const activeEpoch = activeArtifact
          ? numberField(activeArtifact.status?.epoch)
          : numberField(current?.status?.latest?.epoch);
        const artifactMaximumEpoch = numberField(
          activeArtifact?.status?.maximumEpoch,
        );
        const activeEpochs = activeArtifact
          ? (artifactMaximumEpoch === undefined ? undefined : artifactMaximumEpoch + 1)
          : current?.plan.training?.epochs;
        const hasCurrent = Boolean(activeArtifact ?? current);
        const activeProgress = hasCurrent && activeEpoch !== undefined && activeEpochs
          ? Math.min(1, (activeEpoch + 1) / activeEpochs)
          : 0;
        const bestTrainScore = numberField(current?.status?.latest?.bestTrainScore);
        const activeRun = activeArtifact
          ? byId.get(activeArtifact.id)
          : current;
        matrices.push({
          id: manifest.id,
          label: manifest.label,
          totalRuns: manifest.runIds.length,
          completedRuns,
          failedRuns,
          queuedRuns: Math.max(
            0,
            manifest.runIds.length - completedRuns - failedRuns - (hasCurrent ? 1 : 0),
          ),
          progress: (completedRuns + activeProgress) / manifest.runIds.length,
          controllable,
          pauseRequested,
          paused: pauseRequested && !active && !activeArtifact,
          ...(hasCurrent ? {
            active: {
              id: activeArtifact?.id ?? activeRun!.plan.id,
              label: activeRun?.plan.label ?? activeArtifact!.id,
              stage: activeArtifact?.status?.stage
                ?? activeRun?.status?.stage
                ?? "training",
              ...(activeEpoch === undefined ? {} : { epoch: activeEpoch }),
              ...(activeEpochs === undefined ? {} : { epochs: activeEpochs }),
              ...(bestTrainScore === undefined ? {} : { bestTrainScore }),
            },
          } : {}),
        });
      } catch {
        continue;
      }
    }
    return matrices;
  }

  private async loadArchivedRun(plan: TrainingPlan): Promise<LoadedTrainingPlan> {
    if (!plan.id || !plan.label || !plan.runDir || !plan.datasetDir || !plan.archived) {
      throw new Error(`Invalid archived MLP training run: ${plan.id ?? "unknown"}`);
    }
    const runDir = path.resolve(this.repoRoot, plan.runDir);
    const datasetDir = path.resolve(this.repoRoot, plan.datasetDir);
    const statusFile = path.join(runDir, "state", "status.json");
    const storedStatus = await readOptionalJson<TrainingStatus>(statusFile);
    const running = processIsAlive(storedStatus?.pid)
      && !isTerminalStage(storedStatus?.stage);
    const updatedAt = (running ? validTimestamp(storedStatus?.updatedAt) : undefined)
      ?? plan.archivedAt
      ?? validTimestamp(storedStatus?.updatedAt)
      ?? validTimestamp(storedStatus?.completedAt)
      ?? validTimestamp(storedStatus?.pausedAt);
    return {
      key: `archive/${plan.id}`,
      plan,
      runDir,
      datasetDir,
      statusFile,
      logFiles: await metricLogFiles(runDir),
      finalizeFile: path.join(runDir, "control", "FINALIZE"),
      status: {
        ...storedStatus,
        stage: running ? storedStatus?.stage ?? "training" : "archived",
        ...(updatedAt ? { updatedAt } : {}),
      },
      ...(updatedAt ? { updatedAt } : {}),
      running,
    };
  }

  private async loadPlan(planFile: string): Promise<LoadedTrainingPlan> {
    const plan = JSON.parse(await fs.readFile(planFile, "utf8")) as TrainingPlan;
    return await this.loadStoredPlan(
      plan,
      path.relative(this.repoRoot, planFile).split(path.sep).join("/"),
      planFile,
    );
  }

  private async loadStoredPlan(
    plan: TrainingPlan,
    key: string,
    sourceFile: string,
  ): Promise<LoadedTrainingPlan> {
    const configuredDatasetDir = plan.datasetDir ?? plan.dataset?.datasetDir;
    if (!plan.id || !plan.label || !plan.runDir || !configuredDatasetDir) {
      throw new Error(`Invalid MLP training plan: ${sourceFile}`);
    }
    const runDir = path.resolve(this.repoRoot, plan.runDir);
    const datasetDir = path.resolve(this.repoRoot, configuredDatasetDir);
    const statusFile = path.join(runDir, "state", "status.json");
    const [status, planStat] = await Promise.all([
      readOptionalJson<TrainingStatus>(statusFile),
      fs.stat(sourceFile),
    ]);
    const updatedAt = validTimestamp(status?.updatedAt)
      ?? validTimestamp(status?.completedAt)
      ?? validTimestamp(status?.failedAt)
      ?? validTimestamp(status?.pausedAt)
      ?? planStat.mtime.toISOString();
    return {
      key,
      plan,
      runDir,
      datasetDir,
      statusFile,
      logFiles: await metricLogFiles(runDir),
      finalizeFile: path.join(runDir, "control", "FINALIZE"),
      ...(status ? { status } : {}),
      ...(updatedAt ? { updatedAt } : {}),
      running: processIsAlive(status?.pid) && !isTerminalStage(status?.stage),
    };
  }
}

export class MlpTrainingMatrixControlError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MlpTrainingMatrixControlError";
  }
}

function recordField(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function numberField(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function metricRecord(value: unknown): MlpTrainingComparisonMetricValues | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const source = value as Record<string, unknown>;
  const metric: MlpTrainingComparisonMetricValues = {};
  for (const key of [
    "normalizedMse",
    "mse",
    "rmse",
    "mae",
    "zeroBaselineMse",
    "mseSkillVsZero",
    "directionAccuracy",
    "correlation",
  ] as const) {
    const number = numberField(source[key]);
    if (number !== undefined) metric[key] = number;
  }
  const rawMse = numberField(source.rawMse);
  const rawRmse = numberField(source.rawRmse);
  const rawMae = numberField(source.rawMae);
  const endpointCorrelation = numberField(source.endpointCorrelation)
    ?? numberField(source.meanHorizonCorrelation);
  if (metric.mse === undefined && rawMse !== undefined) metric.mse = rawMse;
  if (metric.rmse === undefined && rawRmse !== undefined) metric.rmse = rawRmse;
  if (metric.mae === undefined && rawMae !== undefined) metric.mae = rawMae;
  if (metric.correlation === undefined && endpointCorrelation !== undefined) {
    metric.correlation = endpointCorrelation;
  }
  return Object.keys(metric).length > 0 ? metric : undefined;
}

function distributionRecord(
  value: unknown,
): MlpTrainingComparisonDistributionValues | undefined {
  const source = recordField(value);
  if (!source) return undefined;
  const result: MlpTrainingComparisonDistributionValues = {};
  for (const key of [
    "negativeLogLikelihood",
    "unitNegativeLogLikelihood",
    "bitsPerExample",
    "globalBaselineNegativeLogLikelihood",
    "nllImprovementVsGlobal",
    "rawNegativeLogLikelihood",
    "nllImprovementVsRaw",
  ] as const) {
    const number = numberField(source[key]);
    if (number !== undefined) result[key] = number;
  }
  const expectation = metricRecord(source.expectation);
  const perLeadExpectation = Array.isArray(source.perLeadExpectation)
    ? source.perLeadExpectation.flatMap((value) => {
        const metric = metricRecord(value);
        return metric ? [metric] : [];
      })
    : undefined;
  const mode = metricRecord(source.mode);
  if (expectation) result.expectation = expectation;
  if (perLeadExpectation?.length) {
    result.perLeadExpectation = perLeadExpectation;
  }
  if (mode) result.mode = mode;
  return Object.keys(result).length > 0 ? result : undefined;
}

function autoregressiveEpisodeRecord(
  value: unknown,
): MlpTrainingAutoregressiveEpisodeValues | undefined {
  const source = recordField(value);
  if (!source) return undefined;
  const result: MlpTrainingAutoregressiveEpisodeValues = {};
  for (const key of [
    "episodes",
    "sourceEpisodeSeconds",
    "activeCandles",
    "episodeReturnCorrelation",
    "episodeCumulativePathCorrelation",
  ] as const) {
    const number = numberField(source[key]);
    if (number !== undefined) result[key] = number;
  }
  const activeSource = recordField(source.activeCandlesPerEpisode);
  if (activeSource) {
    const minimum = numberField(activeSource.minimum);
    const mean = numberField(activeSource.mean);
    const maximum = numberField(activeSource.maximum);
    if (minimum !== undefined || mean !== undefined || maximum !== undefined) {
      result.activeCandlesPerEpisode = {
        ...(minimum === undefined ? {} : { minimum }),
        ...(mean === undefined ? {} : { mean }),
        ...(maximum === undefined ? {} : { maximum }),
      };
    }
  }
  const pooledCandles = metricRecord(source.pooledCandles);
  const episodeAverage = metricRecord(source.episodeAverage);
  const episodeEndpoint = metricRecord(source.episodeEndpoint);
  if (pooledCandles) result.pooledCandles = pooledCandles;
  if (episodeAverage) result.episodeAverage = episodeAverage;
  if (episodeEndpoint) result.episodeEndpoint = episodeEndpoint;
  const estimatorSource = recordField(source.estimator);
  if (estimatorSource) {
    const estimator: NonNullable<
      MlpTrainingAutoregressiveEpisodeValues["estimator"]
    > = {};
    for (const key of [
      "trajectories",
      "randomizedReplicates",
      "trajectoriesPerReplicate",
      "returnTrajectoryVarianceMean",
      "endpointTrajectoryVarianceMean",
    ] as const) {
      const number = numberField(estimatorSource[key]);
      if (number !== undefined) estimator[key] = number;
    }
    for (const key of [
      "returnMean",
      "cumulativeLogPricePathMean",
      "endpointMean",
    ] as const) {
      const summary = estimatorVarianceRecord(estimatorSource[key]);
      if (summary) estimator[key] = summary;
    }
    if (Object.keys(estimator).length > 0) result.estimator = estimator;
  }
  const likelihoodSource = recordField(source.pathLikelihood);
  if (likelihoodSource) {
    const likelihood: NonNullable<
      MlpTrainingAutoregressiveEpisodeValues["pathLikelihood"]
    > = {};
    const exactMass = numberField(likelihoodSource.exactPathProbabilityMass);
    const episodes = numberField(likelihoodSource.episodes);
    if (exactMass !== undefined) likelihood.exactPathProbabilityMass = exactMass;
    if (episodes !== undefined) likelihood.episodes = episodes;
    for (const key of [
      "realizedNegativeLogDensityPerCandle",
      "realizedBitsPerCandle",
      "sampledPathLogDensityPercentile",
      "twoSidedTypicality",
      "logDensityZScoreVsSampledPaths",
      "perCandleDensityRatioVsSampleMedian",
    ] as const) {
      const summary = likelihoodSummaryRecord(likelihoodSource[key]);
      if (summary) likelihood[key] = summary;
    }
    if (Object.keys(likelihood).length > 0) {
      result.pathLikelihood = likelihood;
    }
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function estimatorVarianceRecord(
  value: unknown,
): MlpTrainingEstimatorVarianceValues | undefined {
  const source = recordField(value);
  if (!source) return undefined;
  const result: MlpTrainingEstimatorVarianceValues = {};
  for (const key of [
    "meanVariance",
    "meanStandardError",
    "p95StandardError",
    "maximumStandardError",
  ] as const) {
    const number = numberField(source[key]);
    if (number !== undefined) result[key] = number;
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function likelihoodSummaryRecord(
  value: unknown,
): MlpTrainingLikelihoodSummaryValues | undefined {
  const source = recordField(value);
  if (!source) return undefined;
  const result: MlpTrainingLikelihoodSummaryValues = {};
  for (const key of [
    "mean",
    "median",
    "p95",
    "minimum",
    "fractionBelow1Percent",
    "fractionBelow5Percent",
  ] as const) {
    const number = numberField(source[key]);
    if (number !== undefined) result[key] = number;
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function outputCalibrationRecord(
  value: unknown,
): MlpTrainingOutputCalibrationValues | undefined {
  const source = recordField(value);
  if (!source) return undefined;
  const transforms = recordField(source.transforms);
  const scaleOnly = recordField(transforms?.scaleOnly);
  const affine = recordField(transforms?.affine);
  const calibrationRaw = metricRecord(source.calibrationRaw);
  const validation = recordField(source.validation);
  const test = recordField(source.test);
  const densityTemperature = recordField(source.densityTemperature);
  const result: MlpTrainingOutputCalibrationValues = {};
  const scale = numberField(scaleOnly?.scale);
  const affineScale = numberField(affine?.scale);
  const affineIntercept = numberField(affine?.intercept);
  const temperature = numberField(densityTemperature?.temperature);
  if (scale !== undefined) result.scale = scale;
  if (affineScale !== undefined) result.affineScale = affineScale;
  if (affineIntercept !== undefined) result.affineIntercept = affineIntercept;
  if (calibrationRaw?.correlation !== undefined) {
    result.calibrationCorrelation = calibrationRaw.correlation;
  }
  const scaleValidation = metricRecord(validation?.scaleOnly);
  const scaleTest = metricRecord(test?.scaleOnly);
  const affineValidation = metricRecord(validation?.affine);
  const affineTest = metricRecord(test?.affine);
  const densityValidation = distributionRecord(densityTemperature?.validation);
  const densityTest = distributionRecord(densityTemperature?.test);
  if (scaleValidation) result.validation = scaleValidation;
  if (scaleTest) result.test = scaleTest;
  if (affineValidation) result.affineValidation = affineValidation;
  if (affineTest) result.affineTest = affineTest;
  if (temperature !== undefined) result.densityTemperature = temperature;
  if (densityValidation) result.densityValidation = densityValidation;
  if (densityTest) result.densityTest = densityTest;
  return Object.keys(result).length > 0 ? result : undefined;
}

async function metricLogFiles(runDir: string): Promise<string[]> {
  const history = path.join(runDir, "logs", "training.history.jsonl");
  const active = await firstExisting([
    path.join(runDir, "logs", "training.jsonl"),
    path.join(runDir, "logs", "runner.log"),
    path.join(runDir, "logs", "training.log"),
  ]);
  return await exists(history) ? [history, active] : [active];
}

async function firstExisting(files: readonly string[]): Promise<string> {
  for (const file of files) {
    try {
      await fs.access(file);
      return file;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return files[0]!;
}

function compareRuns(left: LoadedTrainingPlan, right: LoadedTrainingPlan): number {
  if (left.running !== right.running) return left.running ? -1 : 1;
  const updatedDifference = Date.parse(right.updatedAt ?? "") - Date.parse(left.updatedAt ?? "");
  if (Number.isFinite(updatedDifference) && updatedDifference !== 0) return updatedDifference;
  return left.plan.label.localeCompare(right.plan.label);
}

function validTimestamp(value: string | undefined): string | undefined {
  if (!value) return undefined;
  return Number.isFinite(Date.parse(value)) ? value : undefined;
}

function isTerminalStage(stage: string | undefined): boolean {
  return stage === "complete"
    || stage === "failed"
    || stage === "paused"
    || stage === "cancelled";
}

async function readMetricLogs(files: readonly string[], requestedCursor: number): Promise<{
  cursor: number;
  reset: boolean;
  events: MlpTrainingMetricEvent[];
}> {
  const segments: Array<{ file: string; size: number }> = [];
  for (const file of files) {
    try {
      segments.push({ file, size: (await fs.stat(file)).size });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  const size = segments.reduce((total, segment) => total + segment.size, 0);
  if (segments.length === 0) {
    return { cursor: 0, reset: requestedCursor > 0, events: [] };
  }
  const validCursor = Number.isSafeInteger(requestedCursor) && requestedCursor >= 0
    ? requestedCursor
    : 0;
  let start = validCursor <= size ? validCursor : 0;
  let reset = start !== validCursor;
  const historyBytes = segments.length > 1 ? segments[0]!.size : 0;
  if (validCursor > 0 && validCursor < historyBytes) {
    // A client that was already polling the active segment before history was
    // attached has a cursor in the wrong coordinate space. Reload all series once.
    start = 0;
    reset = true;
  }
  if (size - start > MAX_LOG_CHUNK_BYTES) {
    start = size - MAX_LOG_CHUNK_BYTES;
    reset = true;
  }
  if (start === size) return { cursor: size, reset, events: [] };

  const chunks: Buffer[] = [];
  let segmentOffset = 0;
  for (const segment of segments) {
    const segmentEnd = segmentOffset + segment.size;
    if (segmentEnd > start) {
      const localStart = Math.max(0, start - segmentOffset);
      const descriptor = await fs.open(segment.file, "r");
      try {
        const buffer = Buffer.allocUnsafe(segment.size - localStart);
        const { bytesRead } = await descriptor.read(
          buffer,
          0,
          buffer.length,
          localStart,
        );
        chunks.push(buffer.subarray(0, bytesRead));
      } finally {
        await descriptor.close();
      }
    }
    segmentOffset = segmentEnd;
  }
  const content = Buffer.concat(chunks);
  const finalNewline = content.lastIndexOf(0x0a);
  if (finalNewline < 0) return { cursor: start, reset, events: [] };
  const complete = content.subarray(0, finalNewline + 1).toString("utf8");
  const lines = complete.split("\n");
  if (start > 0 && reset) lines.shift();
  const events: MlpTrainingMetricEvent[] = [];
  for (const line of lines) {
    if (!line.startsWith("{")) continue;
    try {
      const value = JSON.parse(line) as MlpTrainingMetricEvent;
      if (typeof value.event === "string") {
        const canonicalEvent = METRIC_EVENT_ALIASES.get(value.event) ?? value.event;
        if (METRIC_EVENTS.has(canonicalEvent)) {
          events.push(canonicalEvent === value.event
            ? value
            : { ...value, event: canonicalEvent, sourceEvent: value.event });
        }
      }
    } catch {
      // Runner diagnostics and interrupted final lines remain in the log.
    }
  }
  return { cursor: start + finalNewline + 1, reset, events };
}

async function readOptionalJson<T>(file: string): Promise<T | undefined> {
  try {
    return JSON.parse(await fs.readFile(file, "utf8")) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function exists(file: string): Promise<boolean> {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
}

function processIsAlive(pid: number | undefined): boolean {
  if (typeof pid !== "number" || !Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
