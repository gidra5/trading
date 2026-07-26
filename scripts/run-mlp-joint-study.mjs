import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import {
  pruneCompletedMetricsOnlyTrainingState,
  pruneCompletedTrainingState,
  pruneStudyArtifact,
} from "./mlp-study-retention.mjs";

const SECOND_MS = 1_000;
const MINUTE_MS = 60 * SECOND_MS;
const WAIT_INTERVAL_MS = 30_000;
const SUPERVISOR_HEARTBEAT_MS = positiveEnvironmentInteger(
  "TRADING_MLP_SUPERVISOR_HEARTBEAT_MS", 5_000,
);
const TRAINING_STALL_TIMEOUT_MS = positiveEnvironmentInteger(
  "TRADING_MLP_TRAINING_STALL_TIMEOUT_MS", 120_000,
);
const WATCHDOG_STACK_DUMP_GRACE_MS = positiveEnvironmentInteger(
  "TRADING_MLP_WATCHDOG_STACK_DUMP_GRACE_MS", 5_000,
);
const WATCHDOG_TERMINATION_GRACE_MS = positiveEnvironmentInteger(
  "TRADING_MLP_WATCHDOG_TERMINATION_GRACE_MS", 15_000,
);
const TRAINING_MAX_ATTEMPTS = positiveEnvironmentInteger(
  "TRADING_MLP_TRAINING_MAX_ATTEMPTS", 3,
);
const MIN_LINUX_FREE_BYTES = positiveEnvironmentInteger(
  "TRADING_MLP_MIN_LINUX_FREE_GIB", 5,
) * (1024 ** 3);
const MIN_WINDOWS_FREE_BYTES = positiveEnvironmentInteger(
  "TRADING_MLP_MIN_WINDOWS_FREE_GIB", 5,
) * (1024 ** 3);
const PROGRESS_EVENTS = new Set([
  "observability-ready",
  "training-statistics-cache",
  "time-weighting-ready",
  "training-start",
  "population-training-start",
  "population-group-start",
  "population-train-step",
  "population-epoch",
  "population-group-complete",
  "population-training-complete",
  "baseline",
  "baseline-skipped",
  "train-step",
  "epoch",
  "interrupt",
  "training-study-complete",
  "training-complete",
]);
const METRIC_NAMES = [
  "klDivergence",
  "klDivergenceStdDev",
  "deploymentKlDivergence",
  "deploymentKlDivergenceStdDev",
  "probabilityMse",
  "parameterMse",
  "excessEntropy",
  "oracleMutualInformation",
];
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mlPython = path.join(
  repoRoot,
  ".venv-ml",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const planFile = path.resolve(repoRoot, argument("plan") ?? "ml/training-plan.json");
const basePlan = readJsonRequired(planFile);
const studyKey = argument("study") ?? "lossWeightStudy";
const study = basePlan[studyKey];
const supportedDesigns = new Set(["resolution-vi-half-factorial", "full-grid"]);
if (!study || !supportedDesigns.has(study.design)
  || !Array.isArray(study.terms) || study.terms.length !== 6) {
  throw new Error(
    `${studyKey} requires a supported six-term loss-weight design.`,
  );
}
if (!Array.isArray(study.predictionDelayMinutes) || study.predictionDelayMinutes.length === 0) {
  throw new Error(`${studyKey}.predictionDelayMinutes must contain at least one delay.`);
}
if (!study.datasetDir || study.dataset?.selection !== "one-per-disjoint-regime-then-maximin"
  || !study.dataset.calendarDays || !Array.isArray(study.dataset.utcBlocks)) {
  throw new Error("Joint study requires a stratified reduced dataset configuration.");
}
for (const term of study.terms) {
  if (!(term in basePlan.training.lossWeights)) throw new Error(`Unknown loss term '${term}'.`);
}
const fullGrid = study.design === "full-grid";
const lowScale = fullGrid ? undefined : positiveFinite(study.lowScale, "lowScale");
const centerScale = fullGrid ? undefined : positiveFinite(study.centerScale, "centerScale");
const highScale = fullGrid ? undefined : positiveFinite(study.highScale, "highScale");
const weightScales = fullGrid
  ? validateWeightScales(study.weightScales)
  : [lowScale, centerScale, highScale];
if (!fullGrid && !(lowScale < centerScale && centerScale < highScale)) {
  throw new Error("Loss-weight scales must be strictly ordered low < center < high.");
}
const delays = [...new Set(study.predictionDelayMinutes.map(delayMilliseconds))]
  .sort((left, right) => left - right);
const weightVariants = buildWeightVariants();
validateWeightDesign(weightVariants);
const epochs = positiveInteger(argument("epochs") ?? study.epochs, "epochs");
const patience = positiveInteger(argument("patience") ?? study.patience, "patience");
const selectionMetric = study.selectionMetric ?? "klDivergence";
if (selectionMetric !== "klDivergence") {
  throw new Error("The joint delay/weight study must select checkpoints by validation KL.");
}
if (!delays.includes(0) || !delays.includes(60 * MINUTE_MS)) {
  throw new Error("Joint study must retain both 0- and 60-minute delay controls.");
}
const artifactRetention = study.artifactRetention ?? "all";
if (!["all", "best-per-delay"].includes(artifactRetention)) {
  throw new Error(`${studyKey}.artifactRetention must be 'all' or 'best-per-delay'.`);
}
const reviewAfterDelays = new Set(
  (study.reviewAfterDelayMinutes ?? []).map(delayMilliseconds),
);
const trainingPopulationSize = positiveInteger(
  study.trainingPopulationSize ?? 1,
  `${studyKey}.trainingPopulationSize`,
);
const populationWorkers = nonNegativeInteger(
  study.populationWorkers ?? basePlan.training.workers,
  `${studyKey}.populationWorkers`,
);
if (!fullGrid && trainingPopulationSize !== 1) {
  throw new Error("Population training is currently supported only by full-grid studies.");
}

const productionDatasetDir = path.resolve(repoRoot, basePlan.datasetDir);
const datasetDir = path.resolve(repoRoot, study.datasetDir);
const outputRoot = path.resolve(repoRoot, study.outputDir);
const runDir = path.resolve(repoRoot, study.runDir);
const statusFile = path.join(runDir, "status.json");
const logFile = path.join(runDir, "study.log");
const plansDir = path.join(outputRoot, "plans");
const priorityPlanFile = path.join(plansDir, "priority-dataset.json");
const exampleSelectionFile = path.join(plansDir, "example-selection.json");
const exampleSelection = loadStudyExampleSelection();
const priorityPlan = priorityDatasetPlan(exampleSelection);
if (process.argv.includes("--dry-run")) {
  process.stdout.write(`${JSON.stringify({
    event: "joint-study-design",
    delaysMinutes: delays.map((delay) => delay / MINUTE_MS),
    weightVariants: weightVariants.length,
    plannedRuns: delays.length * weightVariants.length,
    epochs,
    patience,
    studyKey,
    design: study.design,
    artifactRetention,
    trainingPopulationSize,
    populationWorkers,
    reviewAfterDelayMinutes: [...reviewAfterDelays].map((delay) => delay / MINUTE_MS),
    reviewGateAfter: study.promotion?.mode === "automatic" ? null : "reduced-screen",
    terms: study.terms,
    scales: fullGrid
      ? weightScales
      : { low: lowScale, center: centerScale, high: highScale },
    dataset: {
      directory: datasetDir,
      days: splitCounts(exampleSelection.days),
      utcBlocks: exampleSelection.utcBlocks,
      examples: selectedExampleCounts(exampleSelection),
      componentCoveragePredictionDelaysMs:
        exampleSelection.componentCoveragePredictionDelaysMs,
    },
  })}\n`);
  process.exit(0);
}
fs.mkdirSync(runDir, { recursive: true });
fs.mkdirSync(plansDir, { recursive: true });
atomicWrite(exampleSelectionFile, `${JSON.stringify(exampleSelection, null, 2)}\n`);
atomicWrite(priorityPlanFile, `${JSON.stringify(priorityPlan, null, 2)}\n`);

const previous = readJson(statusFile);
if (previous?.pid && processIsAlive(previous.pid)
  && !["complete", "failed", "paused", "review"].includes(previous.stage)) {
  throw new Error(`Joint delay/weight study is already running as PID ${previous.pid}.`);
}

let activeChild;
let activeChildGroup = false;
let interruptionSignal;
let windowsSleepInhibitor;
let windowsSleepHeartbeatFile;
let status = {
  planId: basePlan.id,
  studyKey,
  design: study.design,
  artifactRetention,
  componentStoreId: basePlan.componentStoreId,
  pid: process.pid,
  stage: "waiting-for-components",
  startedAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  outputDir: outputRoot,
  logFile,
  delaysMinutes: delays.map((delay) => delay / MINUTE_MS),
  weightVariants: weightVariants.length,
  variants: delays.length * weightVariants.length,
  epochs,
  patience,
  reviewGateAfter: study.promotion?.mode === "automatic" ? null : "reduced-screen",
  studyDatasetDir: datasetDir,
  selectedDays: splitCounts(exampleSelection.days),
  selectedExamples: selectedExampleCounts(exampleSelection),
};
writeStatus();
startWindowsSleepInhibitor();

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    interruptionSignal ??= signal;
    if (activeChild && !activeChild.killed) signalChild(activeChild, signal, activeChildGroup);
  });
}

try {
  status = {
    ...status,
    stage: "priority-dataset",
    updatedAt: new Date().toISOString(),
    message: "Preparing the stratified study subset before the remaining production days.",
  };
  writeStatus();
  await runStage("priority-dataset", process.execPath, [
    path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
    path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
    "--plan", priorityPlanFile,
  ]);
  const preparedManifest = readJsonRequired(path.join(datasetDir, "dataset.json"));
  const automaticPromotion = study.promotion?.mode === "automatic";
  const resumeProductionAfterPriority = study.resumeProductionPreparationAfterPriority !== false;
  const productionResume = resumeProductionAfterPriority
    ? resumeProductionPreparation()
    : {
        stage: "paused-for-review",
        resumed: false,
        message: "Production preparation remains paused by study configuration.",
      };
  status = {
    ...status,
    productionPreparation: productionResume,
    updatedAt: new Date().toISOString(),
  };
  writeStatus();
  const computedFingerprint = createHash("sha256").update(JSON.stringify({
    componentStoreId: basePlan.componentStoreId,
    featureSchemaVersion: preparedManifest.featureSchemaVersion,
    componentFingerprint: fingerprintComponents(preparedManifest),
    exampleSelection,
    implementationFingerprint: fingerprintFiles([
      "ml/train_mlp.py",
      "ml/train_mlp_population.py",
      "ml/population_mlp.py",
      "ml/mlp_model.py",
      "scripts/build-mlp-dataset.ts",
      "apps/server/src/mlp-feature-store.ts",
      "packages/bot-algo/src/mlp-exposure-predictor.ts",
    ]),
    delays,
    weightDesign: {
      design: study.design,
      terms: study.terms,
      ...(fullGrid
        ? { weightScales }
        : { lowScale, centerScale, highScale }),
    },
    training: basePlan.training,
    epochs,
    patience,
    selectionMetric,
  })).digest("hex").slice(0, 12);
  const requestedResumeFingerprint = argument("resume-fingerprint");
  if (requestedResumeFingerprint
    && !/^[a-f0-9]{12}$/u.test(requestedResumeFingerprint)) {
    throw new Error("--resume-fingerprint must be a 12-character lowercase hexadecimal value.");
  }
  const fingerprint = requestedResumeFingerprint ?? computedFingerprint;
  if (requestedResumeFingerprint && requestedResumeFingerprint !== computedFingerprint) {
    const previousSummary = readJson(path.join(outputRoot, "summary.json"));
    if (previousSummary?.experimentFingerprint !== requestedResumeFingerprint) {
      throw new Error(
        `Cannot resume fingerprint ${requestedResumeFingerprint}: `
        + "the existing study summary does not identify that experiment.",
      );
    }
    appendLog(`${JSON.stringify({
      event: "joint-study-fingerprint-resume",
      fingerprint: requestedResumeFingerprint,
      computedFingerprint,
      completedRuns: previousSummary.completedRuns,
    })}\n`);
    status = {
      ...status,
      experimentFingerprint: requestedResumeFingerprint,
      computedFingerprint,
      resumeFingerprintOverride: true,
      updatedAt: new Date().toISOString(),
      message: `Resuming the existing ${requestedResumeFingerprint} experiment lineage.`,
    };
    writeStatus();
  }
  const results = [];
  const resultKeys = new Set();
  const reusableResults = loadReusableResults();
  const statisticsDir = path.join(outputRoot, "statistics");
  fs.mkdirSync(statisticsDir, { recursive: true });
  let stoppedForReview = false;
  for (let delayIndex = 0; delayIndex < delays.length; delayIndex += 1) {
    if (interruptionSignal) throw new Error(`Interrupted by ${interruptionSignal}.`);
    const delayMs = delays[delayIndex];
    const key = delayKey(delayMs);
    const variant = variantPlan(delayMs, key);
    const variantPlanFile = path.join(plansDir, `${key}.json`);
    const featureStatisticsCache = path.join(
      statisticsDir, `features-${fingerprint}.npz`,
    );
    const targetStatisticsCache = path.join(
      statisticsDir, `${key}-targets-${fingerprint}.npz`,
    );
    atomicWrite(variantPlanFile, `${JSON.stringify(variant, null, 2)}\n`);
    let delayWinner;
    if (artifactRetention === "best-per-delay") {
      const completed = [];
      for (const weightVariant of weightVariants) {
        const directory = variantDirectory(key, weightVariant, fingerprint);
        const resultFile = path.join(directory, "study.json");
        let existing = readJson(resultFile);
        if (!validResult(existing, variant, delayMs, weightVariant.weights)) {
          existing = importReusableResult({
            reusableResults,
            delayMs,
            weightVariant,
            variant,
            variantPlanFile,
            directory,
            resultFile,
          });
        }
        if (!validResult(existing, variant, delayMs, weightVariant.weights)) continue;
        const item = studyResultItem(key, delayMs, weightVariant, existing, directory);
        completed.push(item);
        addResult(results, resultKeys, item);
      }
      delayWinner = bestResult(completed);
      if (delayWinner) {
        await reconcileBestPerDelayArtifacts({
          entries: completed,
          winner: delayWinner,
          dataset: datasetDir,
          plan: variant,
          planFile: variantPlanFile,
          targetStatisticsCache,
        });
      }
      if (results.length > 0) writeSummary(results, fingerprint);
    }
    const pending = weightVariants.some((weightVariant) => {
      const resultFile = variantResultFile(key, weightVariant, fingerprint);
      return !validResult(readJson(resultFile), variant, delayMs, weightVariant.weights);
    });
    if (pending) {
      status = {
        ...status,
        stage: "pairing",
        delayIndex: delayIndex + 1,
        delays: delays.length,
        delayMs,
        delayMinutes: delayMs / MINUTE_MS,
        variantKey: key,
        updatedAt: new Date().toISOString(),
        message: `Pairing the prepared inputs with ${delayMs / MINUTE_MS}m-delay oracle rows.`,
      };
      writeStatus();
      await runStage("pairing", process.execPath, [
        path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
        path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
        "--plan", variantPlanFile,
      ]);
    }
    if (pending && fullGrid) {
      const specification = populationTrainingSpecification({
        key,
        fingerprint,
        variant,
        variantPlanFile,
        featureStatisticsCache,
        targetStatisticsCache,
      });
      const jobsDirectory = path.join(runDir, "population-jobs");
      fs.mkdirSync(jobsDirectory, { recursive: true });
      const jobsFile = path.join(jobsDirectory, `${key}.json`);
      atomicWrite(jobsFile, `${JSON.stringify(specification, null, 2)}\n`);
      status = {
        ...status,
        stage: "population-training",
        trainingPopulationSize,
        populationJobs: specification.jobs.length,
        updatedAt: new Date().toISOString(),
        message: `Training the ${key} exhaustive grid in vectorized CUDA populations.`,
      };
      writeStatus();
      await runPopulationTrainingBatch(jobsFile);
    }
    if (fullGrid) {
      const completed = weightVariants.map((weightVariant) => {
        const directory = variantDirectory(key, weightVariant, fingerprint);
        const result = readJsonRequired(path.join(directory, "study.json"));
        if (!validResult(result, variant, delayMs, weightVariant.weights)) {
          throw new Error(
            `Joint variant ${key}/${weightVariant.key} produced an incompatible result.`,
          );
        }
        return studyResultItem(key, delayMs, weightVariant, result, directory);
      });
      delayWinner = bestResult(completed);
      if (artifactRetention === "best-per-delay") {
        await reconcileBestPerDelayArtifacts({
          entries: completed,
          winner: delayWinner,
          dataset: datasetDir,
          plan: variant,
          planFile: variantPlanFile,
          targetStatisticsCache,
        });
      } else {
        for (const item of completed) {
          await ensureAndVerifyStudyArtifact({
            item,
            dataset: datasetDir,
            plan: variant,
            planFile: variantPlanFile,
            targetStatisticsCache,
          });
          logRetention(pruneCompletedTrainingState(item.directory),
            "study-training-state-pruned");
        }
      }
      for (const item of completed) addResult(results, resultKeys, item);
      writeSummary(results, fingerprint);
      if (reviewAfterDelays.has(delayMs)
        && !process.argv.includes("--continue-after-review")) {
        stoppedForReview = true;
        break;
      }
      continue;
    }
    for (let weightIndex = 0; weightIndex < weightVariants.length; weightIndex += 1) {
      const weightVariant = weightVariants[weightIndex];
      const globalIndex = delayIndex * weightVariants.length + weightIndex;
      const directory = variantDirectory(key, weightVariant, fingerprint);
      const resultFile = path.join(directory, "study.json");
      const existing = readJson(resultFile);
      status = {
        ...status,
        stage: "training",
        variant: globalIndex + 1,
        variants: delays.length * weightVariants.length,
        delayIndex: delayIndex + 1,
        delays: delays.length,
        delayMs,
        delayMinutes: delayMs / MINUTE_MS,
        weightVariant: weightIndex + 1,
        weightVariants: weightVariants.length,
        weightVariantKey: weightVariant.key,
        weights: weightVariant.weights,
        updatedAt: new Date().toISOString(),
        message: `Training ${weightVariant.key} at ${delayMs / MINUTE_MS}m delay.`,
      };
      writeStatus();
      if (!validResult(existing, variant, delayMs, weightVariant.weights)) {
        fs.mkdirSync(directory, { recursive: true });
        await runStage("training", mlPython, [
          path.join(repoRoot, "ml/run_with_observability.py"),
          path.join(repoRoot, "ml/train_mlp.py"),
          ...trainingArguments({
            directory,
            resultFile,
            variant,
            variantPlanFile,
            weightVariant,
            featureStatisticsCache,
            targetStatisticsCache,
            workers: variant.training.workers,
          }),
        ]);
      } else {
        appendLog(`${JSON.stringify({
          event: "joint-study-resume", delay: key, weights: weightVariant.key, resultFile,
        })}\n`);
      }
      const result = readJsonRequired(resultFile);
      if (!validResult(result, variant, delayMs, weightVariant.weights)) {
        throw new Error(`Joint variant ${key}/${weightVariant.key} produced an incompatible result.`);
      }
      const item = studyResultItem(key, delayMs, weightVariant, result, directory);
      if (artifactRetention === "all") {
        await ensureAndVerifyStudyArtifact({
          item,
          dataset: datasetDir,
          plan: variant,
          planFile: variantPlanFile,
          targetStatisticsCache,
        });
        logRetention(pruneCompletedTrainingState(directory), "study-training-state-pruned");
      } else if (delayWinner?.key === item.key) {
        await ensureAndVerifyStudyArtifact({
          item,
          dataset: datasetDir,
          plan: variant,
          planFile: variantPlanFile,
          targetStatisticsCache,
        });
        logRetention(pruneCompletedTrainingState(directory), "study-training-state-pruned");
      } else if (!delayWinner
        || result.bestValidationMetrics[selectionMetric]
          < delayWinner.result.bestValidationMetrics[selectionMetric]) {
        await ensureAndVerifyStudyArtifact({
          item,
          dataset: datasetDir,
          plan: variant,
          planFile: variantPlanFile,
          targetStatisticsCache: path.join(
            statisticsDir, `${key}-targets-${fingerprint}.npz`,
          ),
        });
        logRetention(pruneCompletedTrainingState(directory), "study-training-state-pruned");
        if (delayWinner) {
          logRetention(
            pruneStudyArtifact(delayWinner.directory),
            "study-nonwinner-artifact-pruned",
          );
        }
        delayWinner = item;
      } else {
        logRetention(
          pruneCompletedMetricsOnlyTrainingState(directory),
          "study-metrics-only-state-pruned",
        );
        logRetention(pruneStudyArtifact(directory), "study-nonwinner-artifact-pruned");
      }
      addResult(results, resultKeys, item);
      writeSummary(results, fingerprint);
    }
    if (reviewAfterDelays.has(delayMs)
      && !process.argv.includes("--continue-after-review")) {
      stoppedForReview = true;
      break;
    }
  }

  if (stoppedForReview) {
    status = {
      ...status,
      stage: "review",
      reviewDelayMinutes: status.delayMinutes,
      reviewReachedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      summaryFile: path.join(outputRoot, "summary.json"),
      message: `Completed the exhaustive ${status.delayMinutes}m block and stopped for review. `
        + "Resume later delays with --continue-after-review.",
    };
    writeStatus();
  } else if (automaticPromotion) {
    const promoted = await runFullDayPromotions(results);
    await runProductionFinalists(promoted);
  }

  if (!stoppedForReview) {
    status = { ...status, stage: "restoring-priority-pairing", updatedAt: new Date().toISOString() };
    writeStatus();
    await runStage("restoring-priority-pairing", process.execPath, [
      path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
      path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
      "--plan", priorityPlanFile,
    ]);
    status = {
      ...status,
      stage: "complete",
      completedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      summaryFile: path.join(outputRoot, "summary.json"),
      message: automaticPromotion
        ? "All configured promotion stages completed; every verified artifact remains available in the UI."
        : "Reduced delay/weight screen complete and stopped at the manual review gate; no full-day or production promotion was started.",
    };
    writeStatus();
  }
} catch (error) {
  status = {
    ...status,
    stage: interruptionSignal ? "paused" : "failed",
    updatedAt: new Date().toISOString(),
    ...(interruptionSignal
      ? { pausedAt: new Date().toISOString(), message: `Paused by ${interruptionSignal}; variants are resumable.` }
      : { failedAt: new Date().toISOString(), error: error instanceof Error ? error.message : String(error) }),
  };
  writeStatus();
  if (!interruptionSignal) throw error;
} finally {
  stopWindowsSleepInhibitor();
}

async function runStage(stage, command, args) {
  const maxAttempts = isTrainingStage(stage) ? TRAINING_MAX_ATTEMPTS : 1;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const outcome = await runStageAttempt(stage, command, args, attempt, maxAttempts);
    if (interruptionSignal) throw new Error(`${stage} interrupted by ${interruptionSignal}.`);
    if (outcome.exitCode === 0) return;
    if (!outcome.watchdogTriggered || attempt >= maxAttempts) {
      const reason = outcome.watchdogTriggered
        ? ` after ${attempt} watchdog attempts`
        : "";
      throw new Error(`${stage} exited with code ${outcome.exitCode}${reason}.`);
    }
    const recovery = {
      event: "stage-watchdog-restart",
      stage,
      attempt,
      nextAttempt: attempt + 1,
      maxAttempts,
      diagnosticFile: outcome.diagnosticFile,
      timestamp: new Date().toISOString(),
    };
    appendLog(`${JSON.stringify(recovery)}\n`);
    status = {
      ...status,
      recovery,
      message: `Watchdog is restarting stalled ${stage} attempt ${attempt}/${maxAttempts}.`,
      updatedAt: new Date().toISOString(),
    };
    writeStatus();
    await wait(2_000);
  }
}

async function runPopulationTrainingBatch(jobsFile) {
  let populationSize = trainingPopulationSize;
  for (;;) {
    try {
      status = {
        ...status,
        effectiveTrainingPopulationSize: populationSize,
        updatedAt: new Date().toISOString(),
      };
      writeStatus();
      await runStage(
        `exhaustive-population-${populationSize}-training`,
        mlPython,
        [
        path.join(repoRoot, "ml/run_with_observability.py"),
        path.join(repoRoot, "ml/train_mlp_population.py"),
        "--jobs", jobsFile,
        "--population-size", String(populationSize),
      ]);
      return;
    } catch (error) {
      if (interruptionSignal || populationSize === 1) throw error;
      const next = Math.max(1, Math.floor(populationSize / 2));
      appendLog(`${JSON.stringify({
        event: "training-population-fallback",
        jobsFile,
        failedPopulationSize: populationSize,
        nextPopulationSize: next,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
      })}\n`);
      populationSize = next;
    }
  }
}

async function runStageAttempt(stage, command, args, attempt, maxAttempts) {
  assertStageStorage(stage);
  const startedAt = new Date().toISOString();
  const observation = {
    stage,
    attempt,
    maxAttempts,
    startedAt,
    lastOutputAt: startedAt,
    lastProgressAt: startedAt,
    watchdogTimeoutMs: stage === "training" ? TRAINING_STALL_TIMEOUT_MS : null,
  };
  status = {
    ...status,
    stage,
    stageStartedAt: startedAt,
    stageAttempt: attempt,
    stageMaxAttempts: maxAttempts,
    child: observation,
    updatedAt: startedAt,
  };
  writeStatus();
  appendLog(`\n[${startedAt}] ${stage} attempt ${attempt}/${maxAttempts}: ${command} ${args.join(" ")}\n`);
  const child = spawn(command, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      ...(stage === "priority-dataset"
        ? { TRADING_MLP_WORKER_JOBS_BEFORE_RECYCLE: "2" }
        : {}),
      ...(stage === "verification" ? { TRADING_MLP_VERIFY_CUDA: "false" } : {}),
    },
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  activeChild = child;
  activeChildGroup = true;
  observation.pid = child.pid;
  status.child = { ...observation };
  writeStatus();

  let watchdogTriggered = false;
  let diagnosticFile;
  let terminationTimer;
  let killTimer;
  consume(child.stdout, false, observation);
  consume(child.stderr, true, observation);
  const heartbeat = setInterval(() => {
    const now = Date.now();
    touchWindowsSleepHeartbeat(now);
    const lastProgress = Date.parse(observation.lastProgressAt);
    const lastOutput = Date.parse(observation.lastOutputAt);
    status = {
      ...status,
      supervisorHeartbeatAt: new Date(now).toISOString(),
      child: {
        ...observation,
        outputSilenceSeconds: Math.max(0, Math.floor((now - lastOutput) / SECOND_MS)),
        noProgressSeconds: Math.max(0, Math.floor((now - lastProgress) / SECOND_MS)),
      },
    };
    writeStatus(false);
    if (watchdogTriggered || !isTrainingStage(stage)
      || now - lastProgress < TRAINING_STALL_TIMEOUT_MS) return;
    watchdogTriggered = true;
    diagnosticFile = captureWatchdogDiagnostics(child, observation);
    const watchdog = {
      event: "stage-watchdog-triggered",
      stage,
      pid: child.pid,
      attempt,
      maxAttempts,
      noProgressSeconds: Math.floor((now - lastProgress) / SECOND_MS),
      diagnosticFile,
      stackDumpSignal: "SIGUSR1",
      timestamp: new Date(now).toISOString(),
    };
    appendLog(`${JSON.stringify(watchdog)}\n`);
    status = {
      ...status,
      lastWatchdog: watchdog,
      message: `No ${stage} progress for ${watchdog.noProgressSeconds}s; collecting stacks before restart.`,
      updatedAt: watchdog.timestamp,
    };
    writeStatus();
    signalChild(child, "SIGUSR1", false);
    terminationTimer = setTimeout(() => {
      signalChild(child, "SIGTERM", true);
    }, WATCHDOG_STACK_DUMP_GRACE_MS);
    killTimer = setTimeout(() => {
      signalChild(child, "SIGKILL", true);
    }, WATCHDOG_STACK_DUMP_GRACE_MS + WATCHDOG_TERMINATION_GRACE_MS);
  }, SUPERVISOR_HEARTBEAT_MS);
  heartbeat.unref();

  const exitCode = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve(code ?? (signal ? 128 : 1)));
  });
  clearInterval(heartbeat);
  clearTimeout(terminationTimer);
  clearTimeout(killTimer);
  activeChild = undefined;
  activeChildGroup = false;
  status = {
    ...status,
    supervisorHeartbeatAt: new Date().toISOString(),
    child: {
      ...status.child,
      exitedAt: new Date().toISOString(),
      exitCode,
      watchdogTriggered,
    },
  };
  writeStatus(false);
  return { exitCode, watchdogTriggered, diagnosticFile };
}

async function verifyStudyArtifact(directory) {
  const manifest = readJson(path.join(directory, "manifest.json"));
  if (manifest?.verification
    && fs.existsSync(path.join(directory, manifest.modelFile ?? "model.onnx"))) return;
  status = {
    ...status,
    stage: "verification",
    artifactDir: directory,
    updatedAt: new Date().toISOString(),
  };
  writeStatus();
  await runStage("verification", process.execPath, [
    path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
    path.join(repoRoot, "scripts/verify-mlp-model.mjs"),
    directory,
  ]);
}

function populationTrainingSpecification({
  key,
  fingerprint,
  variant,
  variantPlanFile,
  featureStatisticsCache,
  targetStatisticsCache,
}) {
  const jobs = weightVariants.map((weightVariant) => {
    const directory = variantDirectory(key, weightVariant, fingerprint);
    const resultFile = path.join(directory, "study.json");
    fs.mkdirSync(directory, { recursive: true });
    return {
      key: weightVariant.key,
      output: directory,
      resultFile,
      modelId: `${variant.id}-${weightVariant.key}`,
      label: `${variant.label} · weights ${weightVariant.key}`,
      lossWeights: weightVariant.weights,
    };
  });
  return {
    version: 1,
    common: {
      dataset: datasetDir,
      plan: variantPlanFile,
      epochs,
      batchSize: variant.training.batchSize,
      evaluationBatchSize:
        variant.training.evaluationBatchSize ?? variant.training.batchSize,
      validationFraction: study.screeningValidationFraction ?? 1,
      accumulate: variant.training.gradientAccumulation,
      learningRate: variant.training.learningRate,
      weightDecay: variant.training.weightDecay,
      dropout: variant.training.dropout,
      statesPerExample: variant.training.statesPerExample,
      patience,
      workers: populationWorkers,
      seed: variant.training.seed,
      device: variant.training.device,
      logEverySteps: variant.training.logEverySteps,
      timeWeighting: variant.training.timeWeighting,
      selectionMetric,
      featureStatisticsCache,
      targetStatisticsCache,
      compile: Boolean(variant.training.compile),
      // Six-epoch screening groups are cheaper to redo than an 800 MiB
      // population checkpoint is to write after every epoch.
      checkpointEveryEpochs: 0,
    },
    jobs,
  };
}

function trainingArguments({
  directory,
  resultFile,
  variant,
  variantPlanFile,
  weightVariant,
  featureStatisticsCache,
  targetStatisticsCache,
  workers,
}) {
  return [
    "--dataset", datasetDir,
    "--output", directory,
    "--model-id", `${variant.id}-${weightVariant.key}`,
    "--label", `${variant.label} · weights ${weightVariant.key}`,
    "--plan", variantPlanFile,
    "--epochs", String(epochs),
    "--batch-size", String(variant.training.batchSize),
    "--evaluation-batch-size", String(
      variant.training.evaluationBatchSize ?? variant.training.batchSize,
    ),
    "--validation-fraction", String(study.screeningValidationFraction ?? 1),
    "--accumulate", String(variant.training.gradientAccumulation),
    "--learning-rate", String(variant.training.learningRate),
    "--weight-decay", String(variant.training.weightDecay),
    "--dropout", String(variant.training.dropout),
    "--states-per-example", String(variant.training.statesPerExample),
    "--patience", String(patience),
    "--workers", String(workers),
    "--seed", String(variant.training.seed),
    "--device", variant.training.device,
    "--log-every-steps", String(variant.training.logEverySteps),
    "--loss-weights-json", JSON.stringify(weightVariant.weights),
    "--time-weighting-json", JSON.stringify(variant.training.timeWeighting),
    "--selection-metric", selectionMetric,
    "--study-file", resultFile,
    "--feature-statistics-cache", featureStatisticsCache,
    "--target-statistics-cache", targetStatisticsCache,
    "--skip-baseline",
    "--resume",
    ...(variant.training.compile ? ["--compile"] : []),
  ];
}

async function ensureStudyArtifact({
  directory,
  dataset,
  plan,
  planFile: artifactPlanFile,
  weightVariant,
  resultFile,
  modelId,
  label,
  targetStatisticsCache,
}) {
  const manifest = readJson(path.join(directory, "manifest.json"));
  if (manifest && fs.existsSync(path.join(directory, manifest.modelFile ?? "model.onnx"))) return;
  status = {
    ...status,
    stage: "artifact-export",
    artifactDir: directory,
    message: "Evaluating the test split and exporting the retained best checkpoint.",
    updatedAt: new Date().toISOString(),
  };
  writeStatus();
  await runStage("artifact-export", mlPython, [
    path.join(repoRoot, "ml/export_mlp_study_artifact.py"),
    "--dataset", dataset,
    "--output", directory,
    "--study-file", resultFile,
    "--target-statistics-cache", targetStatisticsCache,
    "--model-id", modelId,
    "--label", label,
    "--plan", artifactPlanFile,
    "--evaluation-batch-size", String(
      plan.training.evaluationBatchSize ?? plan.training.batchSize,
    ),
    "--states-per-example", String(plan.training.statesPerExample),
    "--workers", String(plan.training.workers),
    "--dropout", String(plan.training.dropout),
    "--seed", String(plan.training.seed),
    "--device", plan.training.device,
    "--selection-metric", selectionMetric,
    "--loss-weights-json", JSON.stringify(weightVariant.weights),
    "--time-weighting-json", JSON.stringify(plan.training.timeWeighting),
  ]);
}

async function ensureAndVerifyStudyArtifact({
  item,
  dataset,
  plan,
  planFile: artifactPlanFile,
  targetStatisticsCache,
}) {
  const manifest = readJson(path.join(item.directory, "manifest.json"));
  if (!manifest && !fs.existsSync(path.join(item.directory, "best-model.pt"))) {
    throw new Error(
      `Winning result ${item.key} has neither a retained artifact nor PyTorch best weights.`,
    );
  }
  await ensureStudyArtifact({
    directory: item.directory,
    dataset,
    plan,
    planFile: artifactPlanFile,
    weightVariant: item.weightVariant,
    resultFile: path.join(item.directory, "study.json"),
    modelId: `${plan.id}-${item.weightVariant.key}`,
    label: `${plan.label} · weights ${item.weightVariant.key}`,
    targetStatisticsCache,
  });
  await verifyStudyArtifact(item.directory);
}

async function reconcileBestPerDelayArtifacts({
  entries,
  winner,
  dataset,
  plan,
  planFile: artifactPlanFile,
  targetStatisticsCache,
}) {
  await ensureAndVerifyStudyArtifact({
    item: winner,
    dataset,
    plan,
    planFile: artifactPlanFile,
    targetStatisticsCache,
  });
  logRetention(
    pruneCompletedTrainingState(winner.directory),
    "study-training-state-pruned",
  );
  for (const entry of entries) {
    if (entry.key === winner.key) continue;
    logRetention(
      pruneCompletedMetricsOnlyTrainingState(entry.directory),
      "study-metrics-only-state-pruned",
    );
    logRetention(
      pruneStudyArtifact(entry.directory),
      "study-nonwinner-artifact-pruned",
    );
  }
}

function studyResultItem(key, delayMs, weightVariant, result, directory) {
  return {
    key: `${key}/${weightVariant.key}`,
    delayKey: key,
    delayMs,
    delayMinutes: delayMs / MINUTE_MS,
    weightVariant,
    result,
    directory,
  };
}

function addResult(results, keys, item) {
  if (keys.has(item.key)) return;
  keys.add(item.key);
  results.push(item);
}

function bestResult(entries) {
  return entries.length === 0 ? undefined : entries.reduce((left, right) =>
    right.result.bestValidationMetrics[selectionMetric]
      < left.result.bestValidationMetrics[selectionMetric] ? right : left);
}

function logRetention(retention, event) {
  if (retention.removedFiles.length === 0) return;
  appendLog(`${JSON.stringify({
    event,
    directory: retention.directory,
    removedFiles: retention.removedFiles,
    removedBytes: retention.removedBytes,
  })}\n`);
}

function loadReusableResults() {
  const sourceKey = study.reuseResultsFrom;
  const sourceStudy = sourceKey ? basePlan[sourceKey] : undefined;
  if (!sourceStudy?.outputDir) return new Map();
  const root = path.resolve(repoRoot, sourceStudy.outputDir);
  const reusable = new Map();
  for (const resultFile of findFiles(root, "study.json")) {
    const result = readJson(resultFile);
    if (!result?.bestValidationMetrics || result.finalizedEarly === true) continue;
    const weights = normalizeLossWeights(result.lossWeights);
    if (!weights) continue;
    const key = reusableResultKey(result.predictionDelayMs, weights);
    const candidate = {
      result,
      directory: path.dirname(resultFile),
      modifiedAt: fs.statSync(resultFile).mtimeMs,
    };
    const previous = reusable.get(key);
    if (!previous || candidate.modifiedAt > previous.modifiedAt) reusable.set(key, candidate);
  }
  return reusable;
}

function importReusableResult({
  reusableResults,
  delayMs,
  weightVariant,
  variant,
  variantPlanFile,
  directory,
  resultFile,
}) {
  const reusable = reusableResults.get(reusableResultKey(delayMs, weightVariant.weights));
  if (!reusable
    || !validResult(reusable.result, variant, delayMs, weightVariant.weights)) return undefined;
  fs.mkdirSync(directory, { recursive: true });
  const imported = {
    ...reusable.result,
    reusedFrom: reusable.directory,
  };
  const sourceManifest = readJson(path.join(reusable.directory, "manifest.json"));
  const sourceModel = sourceManifest
    ? path.join(reusable.directory, sourceManifest.modelFile ?? "model.onnx")
    : undefined;
  if (sourceManifest?.verification && sourceModel && fs.existsSync(sourceModel)) {
    materializeReusableArtifact({
      sourceDirectory: reusable.directory,
      sourceManifest,
      directory,
      modelId: `${variant.id}-${weightVariant.key}`,
      label: `${variant.label} · weights ${weightVariant.key}`,
      planFile: variantPlanFile,
    });
    imported.artifact = path.join(directory, sourceManifest.modelFile ?? "model.onnx");
  } else {
    delete imported.artifact;
  }
  atomicWrite(resultFile, `${JSON.stringify(imported, null, 2)}\n`);
  if (imported.artifact && study.retireReusedArtifacts === true) {
    logRetention(
      pruneStudyArtifact(reusable.directory),
      "study-reused-source-artifact-retired",
    );
  }
  appendLog(`${JSON.stringify({
    event: "joint-study-result-reused",
    delayMs,
    weights: weightVariant.weights,
    source: reusable.directory,
    destination: directory,
    artifactReused: Boolean(imported.artifact),
  })}\n`);
  return imported;
}

function materializeReusableArtifact({
  sourceDirectory,
  sourceManifest,
  directory,
  modelId,
  label,
  planFile: artifactPlanFile,
}) {
  const artifactFiles = [
    sourceManifest.modelFile ?? "model.onnx",
    sourceManifest.verificationFixture?.inputFile,
    sourceManifest.verificationFixture?.outputFile,
  ].filter(Boolean);
  for (const file of artifactFiles) {
    linkOrCopy(path.join(sourceDirectory, file), path.join(directory, file));
  }
  const manifest = {
    ...sourceManifest,
    id: modelId,
    label,
    training: {
      ...sourceManifest.training,
      planFile: artifactPlanFile,
      reusedFrom: sourceDirectory,
    },
  };
  atomicWrite(path.join(directory, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
}

function linkOrCopy(source, destination) {
  if (fs.existsSync(destination)) return;
  try {
    fs.linkSync(source, destination);
  } catch (error) {
    if (error?.code !== "EXDEV") throw error;
    fs.copyFileSync(source, destination);
  }
}

function findFiles(root, name) {
  if (!fs.existsSync(root)) return [];
  const files = [];
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const candidate = path.join(directory, entry.name);
      if (entry.isDirectory()) pending.push(candidate);
      else if (entry.isFile() && entry.name === name) files.push(candidate);
    }
  }
  return files;
}

function normalizeLossWeights(weights) {
  if (!weights || typeof weights !== "object") return undefined;
  const aliases = {
    crossEntropy: "cross_entropy",
    probabilityMse: "probability_mse",
    parameterMse: "parameter_mse",
    excessEntropy: "excess_entropy",
    oracleMutualInformation: "oracle_mutual_information",
  };
  const normalized = {};
  for (const term of study.terms) {
    const value = Number(weights[term] ?? weights[aliases[term]]);
    if (!Number.isFinite(value)) return undefined;
    normalized[term] = value;
  }
  return normalized;
}

function reusableResultKey(delayMs, weights) {
  return `${delayMs}|${study.terms.map((term) => Number(weights[term]).toPrecision(17)).join("|")}`;
}

function resumeProductionPreparation() {
  const productionStatusFile = path.resolve(repoRoot, basePlan.runDir, "status.json");
  const productionStatus = readJson(productionStatusFile);
  if (productionStatus?.stage === "complete") {
    return { stage: "complete", message: "Production components were already complete." };
  }
  if (processIsAlive(productionStatus?.pid)) {
    return { stage: productionStatus.stage, pid: productionStatus.pid, resumed: false };
  }
  const child = spawn(process.execPath, [
    path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
    path.join(repoRoot, "scripts/run-mlp-training.mjs"),
    "--dataset-only",
  ], {
    cwd: repoRoot,
    env: process.env,
    detached: true,
    stdio: "ignore",
  });
  child.unref();
  appendLog(`${JSON.stringify({
    event: "production-preparation-resumed",
    pid: child.pid,
    sourceStudyDataset: datasetDir,
  })}\n`);
  return {
    stage: "starting",
    pid: child.pid,
    resumed: true,
    message: "Production preparation resumed with sparse study components as seeds.",
  };
}

function consume(stream, stderr, observation) {
  const lines = readline.createInterface({ input: stream });
  lines.on("line", (line) => {
    observation.lastOutputAt = new Date().toISOString();
    appendLog(`${line}\n`);
    (stderr ? process.stderr : process.stdout).write(`${line}\n`);
    try {
      const event = JSON.parse(line);
      if (PROGRESS_EVENTS.has(event.event)) {
        observation.lastProgressAt = observation.lastOutputAt;
      }
      status = { ...status, latest: event, updatedAt: new Date().toISOString() };
      if (event.event === "epoch") {
        status = {
          ...status,
          epoch: event.epoch,
          epochs: event.epochs,
          validation: event.validation,
          bestValidation: event.bestValidation,
          bestEpoch: event.bestEpoch,
        };
      }
      if (event.event === "population-epoch") {
        status = {
          ...status,
          populationEpoch: event.epoch,
          populationEpochs: event.epochs,
          populationJobs: event.jobs,
        };
      }
      if (event.event === "population-group-start"
        || event.event === "population-group-complete") {
        status = {
          ...status,
          populationGroup: event.populationGroup,
          populationGroups: event.populationGroups,
          completedPopulationJobs:
            event.completedPopulationJobs ?? status.completedPopulationJobs,
          pendingPopulationJobs:
            event.pendingPopulationJobs ?? status.pendingPopulationJobs,
        };
      }
      status.child = { ...observation };
      writeStatus();
    } catch {
      // Human-readable diagnostics remain in the append-only log.
    }
  });
}

function variantPlan(delayMs, key) {
  return {
    ...priorityPlan,
    id: `${basePlan.id}-study-${key}`,
    label: `${basePlan.label.replace(" · 60-second policy delay", "")} · joint study ${formatDelay(delayMs)}`,
    predictionDelayMs: delayMs,
    artifactDir: path.relative(repoRoot, path.join(outputRoot, key, "artifact")),
    runDir: path.relative(repoRoot, path.join(outputRoot, key, "run")),
  };
}

function priorityDatasetPlan(selection) {
  return {
    ...basePlan,
    id: `${basePlan.id}-study-priority-subset`,
    label: `${basePlan.label} · stratified priority study subset`,
    datasetDir: path.relative(repoRoot, datasetDir),
    artifactDir: path.relative(repoRoot, path.join(outputRoot, "priority-artifact-unused")),
    runDir: path.relative(repoRoot, path.join(outputRoot, "priority-run")),
    exampleSelection: {
      ...selection,
      ...(study.dataset.reuseCompletedProductionComponents
        ? { sourceDatasetDir: path.relative(repoRoot, productionDatasetDir) }
        : {}),
    },
  };
}

async function runFullDayPromotions(screeningResults) {
  const promotion = study.promotion;
  if (!promotion) return screeningResults;
  const finalistsPerDelay = positiveInteger(
    promotion.finalistsPerDelay,
    "promotion.finalistsPerDelay",
  );
  const finalists = delays.flatMap((delayMs) => screeningResults
    .filter((item) => item.delayMs === delayMs)
    .sort((left, right) => left.result.bestValidationMetrics.klDivergence
      - right.result.bestValidationMetrics.klDivergence)
    .slice(0, finalistsPerDelay));
  const fullDatasetDir = path.resolve(repoRoot, promotion.fullDayDatasetDir);
  const fullSelection = {
    ...exampleSelection,
    utcBlocks: [[0, 1_440]],
    sourceDatasetDir: path.relative(repoRoot, datasetDir),
  };
  const fullBasePlan = {
    ...priorityPlan,
    id: `${basePlan.id}-study-full-days`,
    label: `${basePlan.label} · 64-full-day promotion dataset`,
    datasetDir: path.relative(repoRoot, fullDatasetDir),
    exampleSelection: fullSelection,
  };
  const fullBasePlanFile = path.join(plansDir, "promotion-full-days.json");
  atomicWrite(fullBasePlanFile, `${JSON.stringify(fullBasePlan, null, 2)}\n`);
  status = {
    ...status,
    stage: "promotion-full-day-dataset",
    promotionFinalists: finalists.length,
    message: "Extending the sparse screen to the same 64 complete calendar days.",
    updatedAt: new Date().toISOString(),
  };
  writeStatus();
  await runStage("promotion-full-day-dataset", process.execPath, [
    path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
    path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
    "--plan", fullBasePlanFile,
  ]);
  const manifest = readJsonRequired(path.join(fullDatasetDir, "dataset.json"));
  const fingerprint = experimentFingerprint(manifest, fullDatasetDir, {
    stage: "full-days",
    finalists: finalists.map((item) => item.key),
  });
  const promoted = [];
  for (let index = 0; index < finalists.length; index += 1) {
    const finalist = finalists[index];
    const key = delayKey(finalist.delayMs);
    const plan = promotedVariantPlan(fullBasePlan, finalist.delayMs, key, "full-days");
    const planPath = path.join(plansDir, `promotion-full-days-${key}.json`);
    atomicWrite(planPath, `${JSON.stringify(plan, null, 2)}\n`);
    if (index === 0 || finalists[index - 1].delayMs !== finalist.delayMs) {
      await runStage("promotion-full-day-pairing", process.execPath, [
        path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
        path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
        "--plan", planPath,
      ]);
    }
    status = {
      ...status,
      stage: "promotion-full-day-training",
      promotionVariant: index + 1,
      promotionVariants: finalists.length,
      delayMinutes: finalist.delayMinutes,
      weightVariantKey: finalist.weightVariant.key,
      updatedAt: new Date().toISOString(),
    };
    writeStatus();
    const directory = path.join(
      outputRoot,
      "promotions",
      "full-days",
      key,
      `${finalist.weightVariant.key}-e${epochs}-p${patience}-${fingerprint}`,
    );
    const result = await trainPromotion(
      fullDatasetDir,
      directory,
      plan,
      planPath,
      finalist.weightVariant,
      "full-days",
      fingerprint,
    );
    promoted.push({ ...finalist, result, directory, datasetStage: "full-days" });
  }
  atomicWrite(path.join(outputRoot, "promotions", "full-days-summary.json"), `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    finalists: promoted.map(promotionSummaryRow),
  }, null, 2)}\n`);
  return promoted;
}

async function runProductionFinalists(promoted) {
  const count = positiveInteger(
    study.promotion?.productionFinalists ?? 1,
    "promotion.productionFinalists",
  );
  const finalists = [...promoted]
    .sort((left, right) => left.result.bestValidationMetrics.klDivergence
      - right.result.bestValidationMetrics.klDivergence)
    .slice(0, count);
  status = {
    ...status,
    stage: "waiting-for-production-components",
    productionFinalists: finalists.length,
    message: "Waiting for the resumable full production component pass before final confirmation.",
    updatedAt: new Date().toISOString(),
  };
  writeStatus();
  await waitForProductionPreparation();
  const confirmed = [];
  for (let index = 0; index < finalists.length; index += 1) {
    const finalist = finalists[index];
    const key = delayKey(finalist.delayMs);
    const plan = promotedVariantPlan(basePlan, finalist.delayMs, key, "production-final");
    const planPath = path.join(
      plansDir,
      `production-final-${key}-${finalist.weightVariant.key}.json`,
    );
    atomicWrite(planPath, `${JSON.stringify(plan, null, 2)}\n`);
    await runStage("production-final-pairing", process.execPath, [
      path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
      path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
      "--plan", planPath,
    ]);
    const manifest = readJsonRequired(path.join(productionDatasetDir, "dataset.json"));
    const fingerprint = experimentFingerprint(manifest, productionDatasetDir, {
      stage: "production-final",
      delayMs: finalist.delayMs,
      weights: finalist.weightVariant.weights,
    });
    status = {
      ...status,
      stage: "production-final-training",
      productionFinalist: index + 1,
      productionFinalists: finalists.length,
      delayMinutes: finalist.delayMinutes,
      weightVariantKey: finalist.weightVariant.key,
      updatedAt: new Date().toISOString(),
    };
    writeStatus();
    const directory = path.join(
      outputRoot,
      "promotions",
      "production",
      key,
      `${finalist.weightVariant.key}-e${epochs}-p${patience}-${fingerprint}`,
    );
    const result = await trainPromotion(
      productionDatasetDir,
      directory,
      plan,
      planPath,
      finalist.weightVariant,
      "production-final",
      fingerprint,
    );
    confirmed.push({ ...finalist, result, directory, datasetStage: "production-final" });
  }
  await runStage("restoring-production-pairing", process.execPath, [
    path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
    path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
    "--plan", planFile,
  ]);
  atomicWrite(path.join(outputRoot, "promotions", "production-summary.json"), `${JSON.stringify({
    generatedAt: new Date().toISOString(),
    finalists: confirmed.map(promotionSummaryRow),
  }, null, 2)}\n`);
}

async function trainPromotion(
  promotionDatasetDir,
  directory,
  plan,
  planPath,
  weightVariant,
  stage,
  fingerprint,
) {
  const resultFile = path.join(directory, "study.json");
  const existing = readJson(resultFile);
  if (!validResult(existing, plan, plan.predictionDelayMs, weightVariant.weights)) {
    fs.mkdirSync(directory, { recursive: true });
    await runStage(`${stage}-training`, mlPython, [
      path.join(repoRoot, "ml/run_with_observability.py"),
      path.join(repoRoot, "ml/train_mlp.py"),
      "--dataset", promotionDatasetDir,
      "--output", directory,
      "--model-id", `${plan.id}-${weightVariant.key}`,
      "--label", `${plan.label} · weights ${weightVariant.key}`,
      "--plan", planPath,
      "--epochs", String(epochs),
      "--batch-size", String(plan.training.batchSize),
      "--evaluation-batch-size", String(
        plan.training.evaluationBatchSize ?? plan.training.batchSize,
      ),
      "--validation-fraction", "1",
      "--accumulate", String(plan.training.gradientAccumulation),
      "--learning-rate", String(plan.training.learningRate),
      "--weight-decay", String(plan.training.weightDecay),
      "--dropout", String(plan.training.dropout),
      "--states-per-example", String(plan.training.statesPerExample),
      "--patience", String(patience),
      "--workers", String(plan.training.workers),
      "--seed", String(plan.training.seed),
      "--device", plan.training.device,
      "--log-every-steps", String(plan.training.logEverySteps),
      "--loss-weights-json", JSON.stringify(weightVariant.weights),
      "--time-weighting-json", JSON.stringify(plan.training.timeWeighting),
      "--selection-metric", selectionMetric,
      "--study-file", resultFile,
      "--feature-statistics-cache", path.join(
        outputRoot, "statistics", `${stage}-features-${fingerprint}.npz`,
      ),
      "--target-statistics-cache", path.join(
        outputRoot,
        "statistics",
        `${stage}-${delayKey(plan.predictionDelayMs)}-targets-${fingerprint}.npz`,
      ),
      "--skip-baseline",
      "--resume",
      ...(plan.training.compile ? ["--compile"] : []),
    ]);
  }
  await ensureStudyArtifact({
    directory,
    dataset: promotionDatasetDir,
    plan,
    planFile: planPath,
    weightVariant,
    resultFile,
    modelId: `${plan.id}-${weightVariant.key}`,
    label: `${plan.label} · weights ${weightVariant.key}`,
    targetStatisticsCache: path.join(
      outputRoot,
      "statistics",
      `${stage}-${delayKey(plan.predictionDelayMs)}-targets-${fingerprint}.npz`,
    ),
  });
  await verifyStudyArtifact(directory);
  const result = readJsonRequired(resultFile);
  if (!validResult(result, plan, plan.predictionDelayMs, weightVariant.weights)) {
    throw new Error(`Promotion ${stage}/${plan.id}/${weightVariant.key} is incompatible.`);
  }
  const retention = pruneCompletedTrainingState(directory);
  if (retention.removedFiles.length > 0) {
    appendLog(`${JSON.stringify({
      event: "study-training-state-pruned",
      directory,
      removedFiles: retention.removedFiles,
      removedBytes: retention.removedBytes,
    })}\n`);
  }
  return result;
}

function promotedVariantPlan(sourcePlan, delayMs, key, stage) {
  return {
    ...sourcePlan,
    id: `${basePlan.id}-${stage}-${key}`,
    label: `${basePlan.label.replace(" · 60-second policy delay", "")} · ${stage} ${formatDelay(delayMs)}`,
    predictionDelayMs: delayMs,
  };
}

async function waitForProductionPreparation() {
  for (;;) {
    if (interruptionSignal) throw new Error(`Interrupted by ${interruptionSignal}.`);
    const manifest = readJson(path.join(productionDatasetDir, "dataset.json"));
    const productionStatus = readJson(path.resolve(repoRoot, basePlan.runDir, "status.json"));
    if (manifest?.version === 7 && manifest.planId === basePlan.id
      && productionStatus?.stage === "complete") return;
    if (productionStatus && ["failed", "paused"].includes(productionStatus.stage)) {
      throw new Error(`Production preparation is ${productionStatus.stage}; resume it to confirm finalists.`);
    }
    status = {
      ...status,
      stage: "waiting-for-production-components",
      productionPreparation: productionStatus,
      updatedAt: new Date().toISOString(),
    };
    writeStatus();
    await wait(WAIT_INTERVAL_MS);
  }
}

function experimentFingerprint(manifest, root, extra) {
  return createHash("sha256").update(JSON.stringify({
    componentStoreId: basePlan.componentStoreId,
    featureSchemaVersion: manifest.featureSchemaVersion,
    componentFingerprint: fingerprintComponentsAt(manifest, root),
    implementationFingerprint: fingerprintFiles([
      "ml/train_mlp.py",
      "ml/mlp_model.py",
      "scripts/build-mlp-dataset.ts",
    ]),
    training: basePlan.training,
    epochs,
    patience,
    selectionMetric,
    extra,
  })).digest("hex").slice(0, 12);
}

function promotionSummaryRow(item) {
  return {
    delayMinutes: item.delayMinutes,
    weightVariant: item.weightVariant.key,
    weights: item.weightVariant.weights,
    directory: item.directory,
    bestEpoch: item.result.bestEpoch,
    validation: item.result.bestValidationMetrics,
    test: item.result.testMetrics,
  };
}

function buildStudyExampleSelection() {
  const counts = study.dataset.calendarDays;
  for (const split of ["train", "validation", "test"]) {
    if (!Number.isInteger(counts[split]) || counts[split] < 1) {
      throw new Error(`lossWeightStudy.dataset.calendarDays.${split} must be positive.`);
    }
  }
  const utcBlocks = study.dataset.utcBlocks.map(([start, end]) => [Number(start), Number(end)]);
  if (utcBlocks.some(([start, end], index) => !Number.isInteger(start)
    || !Number.isInteger(end) || start < 0 || start >= end || end > 1_440
    || (index > 0 && start < utcBlocks[index - 1][1]))) {
    throw new Error("Study UTC blocks must be sorted, non-overlapping whole-minute ranges.");
  }
  const latestCompleteDay = findLatestCompleteHistoryDay();
  const testEnd = latestCompleteDay + 86_400_000;
  const testStart = testEnd - basePlan.latestTestDays * 86_400_000;
  const { trainRanges, validationRanges } = studyWindowRanges();
  const splitAt = (time) => {
    if (time >= testStart && time < testEnd) return "test";
    if (validationRanges.some((range) => time >= range.start && time < range.end)) {
      return "validation";
    }
    if (trainRanges.some((range) => time >= range.start && time < range.end)) return "train";
    return undefined;
  };
  const first = Math.min(testStart, ...trainRanges.map((range) => range.start),
    ...validationRanges.map((range) => range.start));
  const last = Math.max(testEnd, ...trainRanges.map((range) => range.end),
    ...validationRanges.map((range) => range.end));
  const candidates = { train: [], validation: [], test: [] };
  for (let day = utcDay(first); day < last; day += 86_400_000) {
    for (const split of Object.keys(candidates)) {
      const eligible = utcBlocks.every(([start, end]) =>
        splitAt(day + start * MINUTE_MS + 999) === split
        && splitAt(day + end * MINUTE_MS - 1) === split);
      if (eligible) candidates[split].push(day);
    }
  }
  const ranges = {
    train: mergeRanges(trainRanges),
    validation: mergeRanges(validationRanges),
    test: [{ start: testStart, end: testEnd }],
  };
  const days = [];
  for (const split of ["train", "validation", "test"]) {
    const selected = selectStratifiedDays(candidates[split], ranges[split], counts[split]);
    if (selected.length !== counts[split]) {
      throw new Error(`Could select only ${selected.length}/${counts[split]} ${split} study days.`);
    }
    days.push(...selected.map((day) => ({ date: isoDate(day), split })));
  }
  days.sort((left, right) => left.date.localeCompare(right.date));
  return {
    mode: "explicit-time-blocks",
    days,
    utcBlocks,
    componentCoveragePredictionDelaysMs: delays,
    selection: study.dataset.selection,
  };
}

function loadStudyExampleSelection() {
  const configured = [
    readJson(exampleSelectionFile),
    readJson(path.join(datasetDir, "dataset.json"))?.exampleSelection,
  ].find(validStudyExampleSelection);
  return configured ?? buildStudyExampleSelection();
}

function validStudyExampleSelection(selection) {
  if (!selection || selection.mode !== "explicit-time-blocks"
    || selection.selection !== study.dataset.selection
    || !Array.isArray(selection.days)
    || !Array.isArray(selection.utcBlocks)
    || !Array.isArray(selection.componentCoveragePredictionDelaysMs)) return false;
  const counts = splitCounts(selection.days);
  return ["train", "validation", "test"].every((split) =>
    counts[split] === study.dataset.calendarDays[split]
    && selection.days.filter((item) => item.split === split)
      .every((item) => /^\d{4}-\d{2}-\d{2}$/.test(item.date)))
    && JSON.stringify(selection.utcBlocks) === JSON.stringify(study.dataset.utcBlocks)
    && JSON.stringify(selection.componentCoveragePredictionDelaysMs) === JSON.stringify(delays);
}

function studyWindowRanges() {
  const excluded = new Set(basePlan.excludedAggregateWindows);
  const trainRanges = [];
  const validationRanges = [];
  for (const window of basePlan.windows) {
    if (excluded.has(window.id)) continue;
    const start = Date.parse(`${window.start}T00:00:00.000Z`);
    const end = Date.parse(`${window.end}T00:00:00.000Z`) + 86_400_000;
    const midpoint = start + Math.floor(
      (end - start) / basePlan.samplingIntervalMs / 2,
    ) * basePlan.samplingIntervalMs;
    trainRanges.push({ id: window.id, start, end: midpoint });
    validationRanges.push({ id: window.id, start: midpoint, end });
  }
  return { trainRanges, validationRanges };
}

function mergeRanges(ranges) {
  const result = [];
  for (const range of [...ranges].sort((left, right) => left.start - right.start)) {
    const previous = result.at(-1);
    if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
    else result.push({ start: range.start, end: range.end });
  }
  return result;
}

function selectStratifiedDays(candidates, ranges, count) {
  const selected = [];
  const available = new Set(candidates);
  for (const range of ranges) {
    const midpoint = (range.start + range.end) / 2;
    const candidate = [...available]
      .filter((day) => day < range.end && day + 86_400_000 > range.start)
      .sort((left, right) => Math.abs(left + 43_200_000 - midpoint)
        - Math.abs(right + 43_200_000 - midpoint) || left - right)[0];
    if (candidate === undefined) continue;
    selected.push(candidate);
    available.delete(candidate);
    if (selected.length === count) return selected.sort((left, right) => left - right);
  }
  while (selected.length < count && available.size > 0) {
    const candidate = [...available].sort((left, right) => {
      const leftDistance = selected.length === 0 ? Infinity
        : Math.min(...selected.map((value) => Math.abs(value - left)));
      const rightDistance = selected.length === 0 ? Infinity
        : Math.min(...selected.map((value) => Math.abs(value - right)));
      return rightDistance - leftDistance || left - right;
    })[0];
    selected.push(candidate);
    available.delete(candidate);
  }
  return selected.sort((left, right) => left - right);
}

function findLatestCompleteHistoryDay() {
  const root = path.resolve(repoRoot, basePlan.dataDir,
    "historical/spot-btcusdt/btcusdt/1s");
  const dates = fs.readdirSync(root)
    .map((file) => /^(\d{4}-\d{2}-\d{2})\.jsonl(?:\.gz)?$/.exec(file)?.[1])
    .filter(Boolean)
    .sort();
  if (dates.length === 0) throw new Error("No complete one-second history is available.");
  return Date.parse(`${dates.at(-1)}T00:00:00.000Z`);
}

function splitCounts(days) {
  return Object.fromEntries(["train", "validation", "test"].map((split) => [
    split,
    days.filter((item) => item.split === split).length,
  ]));
}

function selectedExampleCounts(selection) {
  const examplesPerDay = selection.utcBlocks.reduce(
    (sum, [start, end]) => sum + (end - start) * MINUTE_MS / basePlan.samplingIntervalMs,
    0,
  );
  return Object.fromEntries(Object.entries(splitCounts(selection.days)).map(([split, days]) => [
    split,
    days * examplesPerDay,
  ]));
}

function utcDay(time) {
  const date = new Date(time);
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

function isoDate(time) {
  return new Date(time).toISOString().slice(0, 10);
}

function fingerprintComponents(manifest) {
  return fingerprintComponentsAt(manifest, datasetDir);
}

function fingerprintComponentsAt(manifest, root) {
  const components = [
    ...(manifest.componentLayout?.inputComponents ?? []),
    ...(manifest.componentLayout?.oracleComponents ?? []),
  ];
  return components.flatMap((component) => [
    component.features,
    component.teacherParameters,
    component.teacherMetrics,
    component.rawOracleProbabilities,
    component.times,
  ].filter(Boolean).map((file) => {
    const stat = fs.statSync(path.resolve(root, file));
    return [file, stat.size, stat.mtimeMs];
  }));
}

function fingerprintFiles(files) {
  const hash = createHash("sha256");
  for (const file of files) {
    hash.update(file);
    hash.update(fs.readFileSync(path.resolve(repoRoot, file)));
  }
  return hash.digest("hex");
}

function buildWeightVariants() {
  if (fullGrid) {
    const variants = [];
    const combinations = weightScales.length ** study.terms.length;
    for (let index = 0; index < combinations; index += 1) {
      let remaining = index;
      const levelIndexes = Array(study.terms.length).fill(0);
      for (let termIndex = study.terms.length - 1; termIndex >= 0; termIndex -= 1) {
        levelIndexes[termIndex] = remaining % weightScales.length;
        remaining = Math.floor(remaining / weightScales.length);
      }
      const scales = Object.fromEntries(study.terms.map((term, termIndex) => [
        term,
        weightScales[levelIndexes[termIndex]],
      ]));
      const isCenter = Object.values(scales).every((scale) => scale === 1);
      variants.push({
        key: `grid-${levelIndexes.join("")}`,
        kind: isCenter ? "center" : "grid",
        signs: null,
        levelIndexes: Object.fromEntries(study.terms.map((term, termIndex) => [
          term,
          levelIndexes[termIndex],
        ])),
        scales,
        weights: scaledWeights(scales),
      });
    }
    return variants;
  }
  const centerScales = Object.fromEntries(study.terms.map((term) => [term, centerScale]));
  const variants = [{
    key: "center",
    kind: "center",
    signs: null,
    scales: centerScales,
    weights: scaledWeights(centerScales),
  }];
  // I = ABCDEF is a 2^(6-1) resolution-VI design. Main effects and all
  // two-factor interactions are mutually unaliased.
  for (let mask = 0; mask < 32; mask += 1) {
    const signs = study.terms.map((_, index) => index < 5
      ? ((mask >>> index) & 1 ? 1 : -1)
      : 0);
    signs[5] = signs.slice(0, 5).reduce((product, sign) => product * sign, 1);
    const scales = Object.fromEntries(study.terms.map((term, index) => [
      term,
      signs[index] < 0 ? lowScale : highScale,
    ]));
    variants.push({
      key: `factorial-${String(mask).padStart(2, "0")}`,
      kind: "factorial",
      signs: Object.fromEntries(study.terms.map((term, index) => [term, signs[index]])),
      scales,
      weights: scaledWeights(scales),
    });
  }
  return variants;
}

function validateWeightDesign(variants) {
  if (fullGrid) {
    const expected = weightScales.length ** study.terms.length;
    if (variants.length !== expected
      || new Set(variants.map((variant) => variant.key)).size !== variants.length) {
      throw new Error(`Full weight grid must contain ${expected} unique combinations.`);
    }
    for (const term of study.terms) {
      for (const scale of weightScales) {
        const count = variants.filter((variant) => variant.scales[term] === scale).length;
        if (count !== expected / weightScales.length) {
          throw new Error(`Weight factor '${term}' is not balanced at scale ${scale}.`);
        }
      }
    }
    return;
  }
  const factorial = variants.filter((variant) => variant.kind === "factorial");
  if (variants.length !== 33 || factorial.length !== 32
    || new Set(variants.map((variant) => variant.key)).size !== variants.length) {
    throw new Error("Resolution-VI design must contain one center and 32 unique corners.");
  }
  for (let left = 0; left < study.terms.length; left += 1) {
    const leftTerm = study.terms[left];
    if (factorial.reduce((sum, variant) => sum + variant.signs[leftTerm], 0) !== 0) {
      throw new Error(`Weight factor '${leftTerm}' is not balanced.`);
    }
    for (let right = left + 1; right < study.terms.length; right += 1) {
      const rightTerm = study.terms[right];
      const correlation = factorial.reduce(
        (sum, variant) => sum + variant.signs[leftTerm] * variant.signs[rightTerm],
        0,
      );
      if (correlation !== 0) {
        throw new Error(`Weight factors '${leftTerm}' and '${rightTerm}' are aliased.`);
      }
    }
  }
}

function scaledWeights(scales) {
  return Object.fromEntries(study.terms.map((term) => [
    term,
    basePlan.training.lossWeights[term] * scales[term],
  ]));
}

function validateWeightScales(values) {
  if (!Array.isArray(values) || values.length !== 4) {
    throw new Error(`${studyKey}.weightScales must contain exactly four levels.`);
  }
  const scales = values.map(Number);
  if (scales.some((value) => !Number.isFinite(value) || value < 0)
    || scales.some((value, index) => index > 0 && value <= scales[index - 1])) {
    throw new Error(`${studyKey}.weightScales must be finite, non-negative, and increasing.`);
  }
  return scales;
}

function variantDirectory(key, weightVariant, fingerprint) {
  return path.join(
    outputRoot,
    key,
    `${weightVariant.key}-e${epochs}-p${patience}-${fingerprint}`,
  );
}

function variantResultFile(key, weightVariant, fingerprint) {
  return path.join(variantDirectory(key, weightVariant, fingerprint), "study.json");
}

function validResult(result, variant, delayMs, weights) {
  return result
    && result.datasetPlanId === variant.id
    && result.componentStoreId === basePlan.componentStoreId
    && result.predictionDelayMs === delayMs
    && result.selectionMetric === selectionMetric
    && result.epochs === epochs
    && result.patience === patience
    && result.seed === basePlan.training.seed
    && result.finalizedEarly === false
    && (sameRecord(result.lossWeights, weights)
      || sameRecord(result.lossWeights, pythonLossWeights(weights)));
}

function writeSummary(results, fingerprint) {
  const byKey = new Map(results.map((item) => [item.key, item]));
  const rows = results.map((item) => ({
    key: item.key,
    delayKey: item.delayKey,
    delayMs: item.delayMs,
    delayMinutes: item.delayMinutes,
    weightVariant: item.weightVariant.key,
    kind: item.weightVariant.kind,
    signs: item.weightVariant.signs,
    scales: item.weightVariant.scales,
    weights: item.weightVariant.weights,
    resultFile: path.join(item.directory, "study.json"),
    bestEpoch: item.result.bestEpoch,
    validation: item.result.bestValidationMetrics,
    deltaValidationKlFromDelayCenter: byKey.get(`${item.delayKey}/center`)
      ? item.result.bestValidationMetrics.klDivergence
        - byKey.get(`${item.delayKey}/center`).result.bestValidationMetrics.klDivergence
      : null,
    deltaValidationKlFromZeroSameWeights: byKey.get(`${delayKey(0)}/${item.weightVariant.key}`)
      ? item.result.bestValidationMetrics.klDivergence
        - byKey.get(`${delayKey(0)}/${item.weightVariant.key}`).result.bestValidationMetrics.klDivergence
      : null,
    deltaValidationKlFromProductionSameWeights:
      byKey.get(`${delayKey(basePlan.predictionDelayMs)}/${item.weightVariant.key}`)
      ? item.result.bestValidationMetrics.klDivergence
        - byKey.get(`${delayKey(basePlan.predictionDelayMs)}/${item.weightVariant.key}`).result.bestValidationMetrics.klDivergence
      : null,
  }));
  const best = rows.reduce((left, right) =>
    right.validation.klDivergence < left.validation.klDivergence ? right : left);
  const delayAnalyses = delays.map((delayMs) => analyzeDelay(rows, delayMs));
  const zeroCenter = rows.find((row) => row.delayMs === 0 && row.kind === "center");
  for (const analysis of delayAnalyses) {
    analysis.deltaCenterKlFromZero = zeroCenter && analysis.center
      ? analysis.center.validation.klDivergence - zeroCenter.validation.klDivergence
      : null;
  }
  const delayWeightInteractions = study.terms.map((term) => {
    const effects = delayAnalyses
      .filter((analysis) => Number.isFinite(analysis.mainEffects?.[term]?.klDivergence))
      .map((analysis) => ({
        delayMinutes: analysis.delayMinutes,
        validationKlEffect: analysis.mainEffects[term].klDivergence,
      }));
    const values = effects.map((item) => item.validationKlEffect);
    return {
      term,
      effects,
      effectRange: values.length > 1 ? Math.max(...values) - Math.min(...values) : null,
    };
  });
  const summary = {
    version: 2,
    basePlanId: basePlan.id,
    componentStoreId: basePlan.componentStoreId,
    experimentFingerprint: fingerprint,
    generatedAt: new Date().toISOString(),
    selectionMetric,
    epochs,
    patience,
    completedRuns: rows.length,
    plannedRuns: delays.length * weightVariants.length,
    analysisComplete: rows.length === delays.length * weightVariants.length,
    delaysMinutes: delays.map((delay) => delay / MINUTE_MS),
    weightDesign: {
      design: study.design,
      variantsPerDelay: weightVariants.length,
      terms: study.terms,
      baseLossWeights: basePlan.training.lossWeights,
      ...(fullGrid
        ? { scales: weightScales }
        : {
            definingRelation: "I=ABCDEF",
            resolution: 6,
            scales: { low: lowScale, center: centerScale, high: highScale },
          }),
    },
    productionDelayMs: basePlan.predictionDelayMs,
    bestObservedRun: best,
    delayAnalyses,
    delayWeightInteractions,
    interpretation: {
      zeroDelay: "forecasting plus feature compression, model approximation, optimization, and teacher-fit error",
      sixtyMinuteDelay: "the complete 3600-second oracle value horizon is historical at prediction time; residual error still includes feature compression, model approximation, optimization, and teacher-fit error",
    },
    results: rows,
  };
  atomicWrite(path.join(outputRoot, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  atomicWrite(path.join(outputRoot, "summary.md"), markdownSummary(summary));
  status = {
    ...status,
    completedRuns: rows.length,
    bestDelayMinutes: best.delayMinutes,
    bestWeightVariant: best.weightVariant,
    bestValidationKl: best.validation.klDivergence,
    summaryFile: path.join(outputRoot, "summary.json"),
    updatedAt: new Date().toISOString(),
  };
  writeStatus();
}

function analyzeDelay(rows, delayMs) {
  const delayRows = rows.filter((row) => row.delayMs === delayMs);
  const center = delayRows.find((row) => row.kind === "center") ?? null;
  if (fullGrid) return analyzeFullGridDelay(delayRows, delayMs, center);
  const factorial = delayRows.filter((row) => row.kind === "factorial");
  const analysisComplete = factorial.length === 32;
  const best = delayRows.length > 0 ? delayRows.reduce((left, right) =>
    right.validation.klDivergence < left.validation.klDivergence ? right : left) : null;
  const mainEffects = Object.fromEntries(study.terms.map((term) => [term,
    Object.fromEntries(METRIC_NAMES.map((metric) => [metric,
      analysisComplete ? contrast(factorial, metric, (row) => row.signs[term], 2) : null,
    ])),
  ]));
  const pairwiseInteractions = [];
  for (let left = 0; left < study.terms.length; left += 1) {
    for (let right = left + 1; right < study.terms.length; right += 1) {
      const leftTerm = study.terms[left];
      const rightTerm = study.terms[right];
      pairwiseInteractions.push({
        terms: [leftTerm, rightTerm],
        metrics: Object.fromEntries(METRIC_NAMES.map((metric) => [metric,
          analysisComplete ? contrast(
            factorial,
            metric,
            (row) => row.signs[leftTerm] * row.signs[rightTerm],
            4,
          ) : null,
        ])),
      });
    }
  }
  pairwiseInteractions.sort((left, right) =>
    Math.abs(right.metrics.klDivergence ?? 0) - Math.abs(left.metrics.klDivergence ?? 0));
  const factorialMeans = Object.fromEntries(METRIC_NAMES.map((metric) => [metric,
    analysisComplete ? metricMean(factorial, metric) : null,
  ]));
  return {
    delayMs,
    delayMinutes: delayMs / MINUTE_MS,
    completedRuns: delayRows.length,
    plannedRuns: weightVariants.length,
    analysisComplete,
    center,
    bestObservedRun: best,
    factorialMetricMeans: factorialMeans,
    curvatureFromCenter: Object.fromEntries(METRIC_NAMES.map((metric) => [
      metric,
      analysisComplete && center
        ? metricDifference(factorialMeans[metric], center.validation[metric])
        : null,
    ])),
    mainEffects,
    pairwiseInteractions,
  };
}

function analyzeFullGridDelay(delayRows, delayMs, center) {
  const analysisComplete = delayRows.length === weightVariants.length;
  const best = delayRows.length > 0 ? delayRows.reduce((left, right) =>
    right.validation[selectionMetric] < left.validation[selectionMetric] ? right : left) : null;
  const levelMarginals = Object.fromEntries(study.terms.map((term) => [
    term,
    weightScales.map((scale) => {
      const matching = delayRows.filter((row) => row.scales[term] === scale);
      return {
        scale,
        weight: basePlan.training.lossWeights[term] * scale,
        runs: matching.length,
        metrics: Object.fromEntries(METRIC_NAMES.map((metric) => [
          metric,
          metricMean(matching, metric),
        ])),
      };
    }),
  ]));
  const mainEffects = Object.fromEntries(study.terms.map((term) => {
    const levels = levelMarginals[term];
    return [term, Object.fromEntries(METRIC_NAMES.map((metric) => [
      metric,
      analysisComplete
        ? metricDifference(levels.at(-1).metrics[metric], levels[0].metrics[metric])
        : null,
    ]))];
  }));
  const pairwiseResponseSurfaces = [];
  if (analysisComplete) {
    for (let left = 0; left < study.terms.length; left += 1) {
      for (let right = left + 1; right < study.terms.length; right += 1) {
        const leftTerm = study.terms[left];
        const rightTerm = study.terms[right];
        pairwiseResponseSurfaces.push({
          terms: [leftTerm, rightTerm],
          cells: weightScales.flatMap((leftScale) => weightScales.map((rightScale) => {
            const matching = delayRows.filter((row) =>
              row.scales[leftTerm] === leftScale && row.scales[rightTerm] === rightScale);
            return {
              leftScale,
              rightScale,
              runs: matching.length,
              metrics: Object.fromEntries(METRIC_NAMES.map((metric) => [
                metric,
                metricMean(matching, metric),
              ])),
            };
          })),
        });
      }
    }
  }
  const gridMeans = Object.fromEntries(METRIC_NAMES.map((metric) => [
    metric,
    metricMean(delayRows, metric),
  ]));
  return {
    delayMs,
    delayMinutes: delayMs / MINUTE_MS,
    completedRuns: delayRows.length,
    plannedRuns: weightVariants.length,
    analysisComplete,
    center,
    bestObservedRun: best,
    gridMetricMeans: gridMeans,
    curvatureFromCenter: Object.fromEntries(METRIC_NAMES.map((metric) => [
      metric,
      analysisComplete && center
        ? metricDifference(gridMeans[metric], center.validation[metric])
        : null,
    ])),
    levelMarginals,
    mainEffects,
    pairwiseResponseSurfaces,
    pairwiseInteractions: [],
  };
}

function contrast(rows, metric, sign, multiplier) {
  if (rows.length === 0 || rows.some((row) => !Number.isFinite(row.validation[metric]))) {
    return null;
  }
  return multiplier * rows.reduce(
    (sum, row) => sum + sign(row) * row.validation[metric],
    0,
  ) / rows.length;
}

function metricMean(rows, metric) {
  const values = rows
    .map((row) => row.validation[metric])
    .filter(Number.isFinite);
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

function metricDifference(left, right) {
  return Number.isFinite(left) && Number.isFinite(right) ? left - right : null;
}

function markdownSummary(summary) {
  const lines = [
    "# MLP joint prediction-delay and loss-weight study",
    "",
    `${summary.completedRuns}/${summary.plannedRuns} combinations complete: ${summary.delaysMinutes.length} delays × ${summary.weightDesign.variantsPerDelay} joint loss-weight settings; ${summary.epochs} epochs, patience ${summary.patience}, selection by validation KL.`,
    "",
    "## Delay summary",
    "",
    "| Delay | Runs | Center KL | Center KL σ | Δ center KL vs 0m | Best KL | Best KL σ | Best weight setting |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
  ];
  for (const analysis of summary.delayAnalyses) {
    lines.push(`| ${formatDelay(analysis.delayMs)} | ${analysis.completedRuns}/${analysis.plannedRuns} | ${number(analysis.center?.validation.klDivergence)} | ${number(analysis.center?.validation.klDivergenceStdDev)} | ${signed(analysis.deltaCenterKlFromZero)} | ${number(analysis.bestObservedRun?.validation.klDivergence)} | ${number(analysis.bestObservedRun?.validation.klDivergenceStdDev)} | ${analysis.bestObservedRun?.weightVariant ?? "—"} |`);
  }
  lines.push(
    "",
    "## Loss-weight main effects by delay",
    "",
    fullGrid
      ? `Each value is marginal average validation KL at scale ${weightScales.at(-1)} `
        + `minus scale ${weightScales[0]}. Negative is better.`
      : "Each value is average validation KL at the high weight minus the low weight. Negative is better. Variation across columns is a delay × weight interaction.",
    "",
    `| Loss term | ${summary.delayAnalyses.map((item) => `${formatDelay(item.delayMs)} Δ KL`).join(" | ")} | Effect range |`,
    `| --- | ${summary.delayAnalyses.map(() => "---:").join(" | ")} | ---: |`,
  );
  for (const interaction of summary.delayWeightInteractions) {
    const effects = new Map(interaction.effects.map((item) => [item.delayMinutes, item.validationKlEffect]));
    lines.push(`| ${interaction.term} | ${summary.delayAnalyses.map((item) => signed(effects.get(item.delayMinutes))).join(" | ")} | ${number(interaction.effectRange)} |`);
  }
  lines.push(
    "",
    "## Best combinations observed",
    "",
    "| Delay | Weight setting | Validation KL | KL σ | pMSE | Parameter MSE | Excess H | Oracle MI |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  );
  for (const row of [...summary.results]
    .sort((left, right) => left.validation.klDivergence - right.validation.klDivergence)
    .slice(0, 20)) {
    const metric = row.validation;
    lines.push(`| ${formatDelay(row.delayMs)} | ${row.weightVariant} | ${number(metric.klDivergence)} | ${number(metric.klDivergenceStdDev)} | ${number(metric.probabilityMse)} | ${number(metric.parameterMse)} | ${number(metric.excessEntropy)} | ${number(metric.oracleMutualInformation)} |`);
  }
  lines.push(
    "",
    fullGrid
      ? "The JSON report retains every run plus all 15 complete 4×4 pairwise marginal response surfaces at every delay."
      : "The JSON report also contains all 15 pairwise loss-weight interactions separately at every delay.",
    "The 60-minute target's complete one-hour oracle value horizon is available in raw history at prediction time; residual error still includes feature compression, optimization, and teacher-fit error.",
    "",
  );
  return `${lines.join("\n")}\n`;
}

function pythonLossWeights(weights) {
  return {
    cross_entropy: weights.crossEntropy,
    probability_mse: weights.probabilityMse,
    parameter_mse: weights.parameterMse,
    excess_entropy: weights.excessEntropy,
    oracle_mutual_information: weights.oracleMutualInformation,
  };
}

function sameRecord(left, right) {
  return left != null && right != null
    && Object.keys(left).length === Object.keys(right).length
    && Object.entries(left).every(([key, value]) => right[key] === value);
}

function delayMilliseconds(value) {
  const delay = Number(value) * MINUTE_MS;
  if (!Number.isSafeInteger(delay) || delay < 0 || delay % SECOND_MS !== 0) {
    throw new Error(`Invalid prediction delay '${value}m'; delays must be non-negative whole seconds.`);
  }
  return delay;
}

function delayKey(delayMs) {
  return `delay-${String(delayMs / SECOND_MS).replace(".", "p")}s`;
}

function formatDelay(delayMs) {
  return `${Number((delayMs / MINUTE_MS).toFixed(6))}m`;
}

function positiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${name} must be a positive integer.`);
  return parsed;
}

function nonNegativeInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  return parsed;
}

function positiveFinite(value, name) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be positive and finite.`);
  }
  return parsed;
}

function positiveEnvironmentInteger(name, fallback) {
  const value = process.env[name];
  if (value === undefined || value === "") return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

function isTrainingStage(stage) {
  return stage === "training" || stage.endsWith("-training");
}

function argument(name) {
  const index = process.argv.indexOf(`--${name}`);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value.`);
  return value;
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function processIsAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function signalChild(child, signal, group) {
  if (!child?.pid) return;
  try {
    process.kill(group ? -child.pid : child.pid, signal);
  } catch (error) {
    if (error?.code !== "ESRCH") throw error;
  }
}

function startWindowsSleepInhibitor() {
  if (!isWsl()) {
    status.sleepInhibitor = {
      active: false,
      reason: "Windows sleep inhibition is only needed under WSL.",
    };
    writeStatus();
    return;
  }
  try {
    const windowsTemp = spawnSync("powershell.exe", [
      "-NoProfile",
      "-Command",
      "[IO.Path]::GetTempPath()",
    ], {
      encoding: "utf8",
      timeout: 10_000,
    }).stdout?.trim();
    if (!windowsTemp) throw new Error("could not resolve the Windows temporary directory");
    const windowsHeartbeat = path.win32.join(
      windowsTemp,
      `trading-mlp-study-${process.pid}.heartbeat`,
    );
    const heartbeatPath = spawnSync("wslpath", ["-u", windowsHeartbeat], {
      encoding: "utf8",
      timeout: 10_000,
    }).stdout?.trim();
    const scriptPath = spawnSync("wslpath", [
      "-w",
      path.join(repoRoot, "scripts/keep-windows-awake.ps1"),
    ], {
      encoding: "utf8",
      timeout: 10_000,
    }).stdout?.trim();
    if (!heartbeatPath || !scriptPath) {
      throw new Error("could not translate Windows sleep-inhibitor paths");
    }
    windowsSleepHeartbeatFile = heartbeatPath;
    touchWindowsSleepHeartbeat(Date.now());
    windowsSleepInhibitor = spawn("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      scriptPath,
      "-HeartbeatFile",
      windowsHeartbeat,
      "-StaleAfterSeconds",
      "180",
      "-PollSeconds",
      "10",
    ], {
      cwd: repoRoot,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    consumeSleepInhibitorOutput(windowsSleepInhibitor.stdout, false);
    consumeSleepInhibitorOutput(windowsSleepInhibitor.stderr, true);
    status.sleepInhibitor = {
      active: true,
      pid: windowsSleepInhibitor.pid,
      heartbeatFile: windowsHeartbeat,
      staleAfterSeconds: 180,
      startedAt: new Date().toISOString(),
    };
    windowsSleepInhibitor.once("exit", (code, signal) => {
      if (status.sleepInhibitor?.pid !== windowsSleepInhibitor?.pid) return;
      status.sleepInhibitor = {
        ...status.sleepInhibitor,
        active: false,
        exitedAt: new Date().toISOString(),
        exitCode: code,
        signal,
      };
      writeStatus();
    });
    writeStatus();
  } catch (error) {
    windowsSleepInhibitor = undefined;
    windowsSleepHeartbeatFile = undefined;
    status.sleepInhibitor = {
      active: false,
      error: error instanceof Error ? error.message : String(error),
    };
    writeStatus();
  }
}

function stopWindowsSleepInhibitor() {
  if (windowsSleepInhibitor?.pid) {
    try {
      windowsSleepInhibitor.kill("SIGTERM");
    } catch {
      // The Windows helper also self-releases after its heartbeat becomes stale.
    }
  }
  if (windowsSleepHeartbeatFile) {
    try {
      fs.unlinkSync(windowsSleepHeartbeatFile);
    } catch (error) {
      if (error?.code !== "ENOENT") {
        appendLog(`${JSON.stringify({
          event: "windows-sleep-inhibitor-cleanup-error",
          error: error instanceof Error ? error.message : String(error),
        })}\n`);
      }
    }
  }
}

function touchWindowsSleepHeartbeat(now) {
  if (!windowsSleepHeartbeatFile) return;
  const timestamp = new Date(now);
  try {
    if (fs.existsSync(windowsSleepHeartbeatFile)) {
      fs.utimesSync(windowsSleepHeartbeatFile, timestamp, timestamp);
    } else {
      fs.writeFileSync(windowsSleepHeartbeatFile, `${timestamp.toISOString()}\n`);
    }
  } catch (error) {
    status.sleepInhibitor = {
      ...status.sleepInhibitor,
      active: false,
      heartbeatError: error instanceof Error ? error.message : String(error),
    };
  }
}

function consumeSleepInhibitorOutput(stream, stderr) {
  const lines = readline.createInterface({ input: stream });
  lines.on("line", (line) => {
    appendLog(`${line}\n`);
    if (stderr) process.stderr.write(`${line}\n`);
    try {
      const event = JSON.parse(line);
      if (event.event === "windows-sleep-inhibitor-ready") {
        status.sleepInhibitor = {
          ...status.sleepInhibitor,
          active: true,
          windowsPid: event.pid,
          readyAt: new Date().toISOString(),
        };
        writeStatus();
      }
    } catch {
      // Human-readable PowerShell diagnostics remain in the study log.
    }
  });
}

function isWsl() {
  return process.platform === "linux"
    && (process.env.WSL_DISTRO_NAME !== undefined
      || fs.existsSync("/proc/sys/fs/binfmt_misc/WSLInterop"));
}

function captureWatchdogDiagnostics(child, observation) {
  const diagnosticsDir = path.join(runDir, "diagnostics");
  fs.mkdirSync(diagnosticsDir, { recursive: true });
  const stamp = new Date().toISOString().replaceAll(/[:.]/g, "-");
  const file = path.join(diagnosticsDir, `${observation.stage}-${child.pid}-${stamp}.json`);
  const tasks = readProcessTasks(child.pid);
  const gpu = spawnSync("nvidia-smi", [], {
    encoding: "utf8",
    timeout: 10_000,
    maxBuffer: 2 * 1024 * 1024,
  });
  const diagnostic = {
    version: 1,
    capturedAt: new Date().toISOString(),
    supervisorPid: process.pid,
    childPid: child.pid,
    observation,
    process: {
      status: readText(`/proc/${child.pid}/status`),
      stat: readText(`/proc/${child.pid}/stat`),
      cmdline: readText(`/proc/${child.pid}/cmdline`)?.replaceAll("\0", " ").trim(),
      tasks,
    },
    gpu: {
      exitCode: gpu.status,
      signal: gpu.signal,
      stdout: gpu.stdout,
      stderr: gpu.stderr,
      error: gpu.error?.message,
    },
    storage: storageStatus(),
  };
  atomicWrite(file, `${JSON.stringify(diagnostic, null, 2)}\n`);
  return file;
}

function readProcessTasks(pid) {
  try {
    return fs.readdirSync(`/proc/${pid}/task`).map((tid) => ({
      tid: Number(tid),
      status: readText(`/proc/${pid}/task/${tid}/status`),
      wchan: readText(`/proc/${pid}/task/${tid}/wchan`)?.trim(),
    }));
  } catch (error) {
    return [{ error: error instanceof Error ? error.message : String(error) }];
  }
}

function readText(file) {
  try {
    return fs.readFileSync(file, "utf8");
  } catch (error) {
    return error?.code === "ENOENT"
      ? undefined
      : `ERROR: ${error instanceof Error ? error.message : String(error)}`;
  }
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return undefined;
    throw error;
  }
}

function readJsonRequired(file) {
  const value = readJson(file);
  if (!value) throw new Error(`Missing JSON file: ${file}`);
  return value;
}

function appendLog(value) {
  fs.appendFileSync(logFile, value);
}

function writeStatus(touchUpdatedAt = true) {
  if (touchUpdatedAt) status.updatedAt = new Date().toISOString();
  status.storage = storageStatus();
  atomicWrite(statusFile, `${JSON.stringify(status, null, 2)}\n`);
}

function storageStatus() {
  return {
    linux: filesystemSpace(repoRoot, MIN_LINUX_FREE_BYTES),
    windowsHost: fs.existsSync("/mnt/c")
      ? filesystemSpace("/mnt/c", MIN_WINDOWS_FREE_BYTES)
      : undefined,
  };
}

function filesystemSpace(directory, minimumFreeBytes) {
  try {
    const stat = fs.statfsSync(directory);
    const freeBytes = stat.bavail * stat.bsize;
    return {
      directory,
      freeBytes,
      freeGiB: freeBytes / (1024 ** 3),
      minimumFreeGiB: minimumFreeBytes / (1024 ** 3),
      healthy: freeBytes >= minimumFreeBytes,
    };
  } catch (error) {
    return {
      directory,
      healthy: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function assertStageStorage(stage) {
  const storage = storageStatus();
  const unhealthy = Object.values(storage).filter((item) => item && !item.healthy);
  if (unhealthy.length === 0) return;
  const details = unhealthy.map((item) => item.error
    ? `${item.directory}: ${item.error}`
    : `${item.directory}: ${number(item.freeGiB)} GiB free, `
      + `${number(item.minimumFreeGiB)} GiB required`).join("; ");
  throw new Error(`Refusing to start ${stage} with unsafe free disk space: ${details}`);
}

function atomicWrite(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, value);
  fs.renameSync(temporary, file);
}

function number(value) {
  return Number.isFinite(value) ? Number(value).toPrecision(7) : "—";
}

function signed(value) {
  if (!Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${Number(value).toPrecision(7)}`;
}
