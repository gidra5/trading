import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const planFile = path.resolve(repoRoot, argument("plan")
  ?? "ml/training-plans/mlp-direct-oracle-temporal-v12-minute-curriculum-60m-to-0m-oracle-mi-half-scratch.json");
const dryRun = process.argv.includes("--dry-run");
const curriculumPlan = readJsonRequired(planFile);
const basePlanFile = path.resolve(repoRoot, curriculumPlan.basePlanFile);
const basePlan = readJsonRequired(basePlanFile);
const curriculum = validateCurriculum(curriculumPlan, basePlan);
const delaysMinutes = delaySchedule(curriculum);
const runDir = path.resolve(repoRoot, curriculumPlan.runDir);
const plansDir = path.join(runDir, "plans");
const phasesRunDir = path.join(runDir, "phases");
const statusFile = path.join(runDir, "status.json");
const summaryFile = path.join(runDir, "curriculum.json");
const logFile = path.join(runDir, "training.log");
const finalizeFile = path.join(runDir, "FINALIZE");
const nativeDir = path.join(runDir, "native");
const nativeLibrary = path.join(
  nativeDir,
  process.platform === "win32" ? "vw_kama_cuda.dll" : "libvw_kama_cuda.so",
);
const runtimeRoot = path.join(repoRoot, "data", "runtime-cache");
const environment = {
  ...process.env,
  PYTHONUTF8: process.env.PYTHONUTF8 || "1",
  PYTHONIOENCODING: process.env.PYTHONIOENCODING || "utf-8",
  TMPDIR: process.env.TRADING_ML_TMP_DIR || path.join(runtimeRoot, "tmp"),
  TEMP: process.env.TRADING_ML_TMP_DIR || path.join(runtimeRoot, "tmp"),
  TMP: process.env.TRADING_ML_TMP_DIR || path.join(runtimeRoot, "tmp"),
  TRITON_CACHE_DIR: process.env.TRITON_CACHE_DIR || path.join(runtimeRoot, "triton"),
  TORCHINDUCTOR_CACHE_DIR:
    process.env.TORCHINDUCTOR_CACHE_DIR || path.join(runtimeRoot, "torchinductor"),
  CUDA_CACHE_PATH: process.env.CUDA_CACHE_PATH || path.join(runtimeRoot, "cuda"),
  VW_KAMA_CUDA_OUTPUT: nativeLibrary,
  VW_KAMA_CUDA_LIBRARY: nativeLibrary,
};

if (dryRun) {
  process.stdout.write(`${JSON.stringify({
    planId: curriculumPlan.id,
    phases: delaysMinutes.length,
    delaysMinutes,
    firstPhaseScratch: true,
    epochsPerPhase: curriculum.epochsPerPhase,
    patience: curriculum.patience,
    targetValidationMetric: "baseKlDivergence",
    targetValidationBaseKl: curriculum.targetValidationBaseKl,
    targetRepresentation: basePlan.training.targetRepresentation,
    samplingIntervalMs: curriculumPlan.samplingIntervalMs,
    datasetRetention: curriculum.datasetRetention,
    modelRetention: "all",
    minimumFreeSpaceGb: curriculum.minimumFreeSpaceGb,
  }, null, 2)}\n`);
  process.exit(0);
}

for (const directory of [
  runDir,
  plansDir,
  phasesRunDir,
  nativeDir,
  ...[
    environment.TMPDIR,
    environment.TRITON_CACHE_DIR,
    environment.TORCHINDUCTOR_CACHE_DIR,
    environment.CUDA_CACHE_PATH,
  ],
]) {
  fs.mkdirSync(directory, { recursive: true });
}

const previousStatus = readJson(statusFile);
if (previousStatus?.pid && processIsAlive(previousStatus.pid)
  && !["complete", "failed", "paused", "cancelled"].includes(previousStatus.stage)) {
  throw new Error(`Delay curriculum is already running as PID ${previousStatus.pid}.`);
}
fs.rmSync(finalizeFile, { force: true });

let summary = readJson(summaryFile) ?? {
  version: 1,
  planId: curriculumPlan.id,
  basePlanFile: relative(basePlanFile),
  delayScheduleMinutes: delaysMinutes,
  epochsPerPhase: curriculum.epochsPerPhase,
  patience: curriculum.patience,
  targetValidationMetric: "baseKlDivergence",
  targetValidationBaseKl: curriculum.targetValidationBaseKl,
  completedPhases: [],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
};
summary = {
  ...summary,
  targetValidationMetric: "baseKlDivergence",
  targetValidationBaseKl: curriculum.targetValidationBaseKl,
  completedPhases: (summary.completedPhases ?? []).map((phase) => ({
    ...phase,
    validationTargetReached: (
      phase.validationTargetReached
      || phase.bestValidation?.baseKlDivergence
        <= curriculum.targetValidationBaseKl
    ),
  })),
};
delete summary.targetValidationKl;
let activeChild;
let activePhaseFinalizeFile;
let interruptionSignal;
let status = {
  planId: curriculumPlan.id,
  pid: process.pid,
  stage: "starting",
  startedAt: previousStatus?.startedAt ?? new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  runDir,
  datasetDir: path.resolve(repoRoot, curriculumPlan.datasetDir),
  artifactDir: path.resolve(repoRoot, curriculumPlan.artifactDir),
  logFile,
  curriculumStage: summary.completedPhases.length + 1,
  curriculumStages: delaysMinutes.length,
  completedCurriculumPhases: summary.completedPhases.length,
  message: "Starting the minute-oracle delay curriculum.",
};
writeStatus();

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    interruptionSignal ??= signal;
    if (activeChild && !activeChild.killed) activeChild.kill(signal);
  });
}

const finalizeForwarder = setInterval(() => {
  if (!activePhaseFinalizeFile || !fs.existsSync(finalizeFile)) return;
  try {
    fs.mkdirSync(path.dirname(activePhaseFinalizeFile), { recursive: true });
    fs.writeFileSync(activePhaseFinalizeFile, `${new Date().toISOString()}\n`);
  } catch {
    // The active phase can finish between the existence check and this write.
  }
}, 1_000);
finalizeForwarder.unref();

try {
  await runStage("feature-contract-build", process.execPath, [
    path.join(repoRoot, "node_modules/typescript/bin/tsc"),
    "-p", path.join(repoRoot, "packages/bot-algo/tsconfig.json"),
  ]);
  await runStage("cuda-build", process.execPath, [
    path.join(repoRoot, "scripts/build-vw-kama-cuda.mjs"),
  ]);

  for (let phaseIndex = 0; phaseIndex < delaysMinutes.length; phaseIndex += 1) {
    const delayMinutes = delaysMinutes[phaseIndex];
    const delayMs = delayMinutes * 60_000;
    const delaySeconds = delayMinutes * 60;
    const phase = phasePaths(delaySeconds);
    const recovered = recoverCompletedPhase(phaseIndex, delayMinutes, phase);
    if (recovered) {
      if (!summary.completedPhases.some((entry) => entry.stage === phaseIndex + 1)) {
        summary.completedPhases.push(recovered);
        writeSummary();
      }
      pruneCompletedPhaseDataset(phaseIndex, phase);
      continue;
    }
    const previousDelaySeconds = phaseIndex === 0
      ? undefined
      : delaysMinutes[phaseIndex - 1] * 60;
    const generatedPlan = generatePhasePlan(
      phaseIndex,
      delayMinutes,
      phase,
      previousDelaySeconds,
    );
    atomicJson(phase.planFile, generatedPlan);
    activePhaseFinalizeFile = path.join(phase.runDir, "FINALIZE");
    clearPhaseMetrics();
    status = {
      ...status,
      stage: "dataset",
      curriculumStage: phaseIndex + 1,
      curriculumStages: delaysMinutes.length,
      completedCurriculumPhases: summary.completedPhases.length,
      delayMinutes,
      predictionDelayMs: delayMs,
      datasetDir: phase.datasetDir,
      artifactDir: phase.artifactDir,
      phasePlanFile: phase.planFile,
      message: (
        `Preparing delay ${delayMinutes}m (${phaseIndex + 1}/${delaysMinutes.length}); `
        + "reusing shared compressed input/oracle components."
      ),
    };
    writeStatus();
    ensureMinimumFreeSpace();
    await runStage("dataset", process.execPath, [
      path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
      path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
      "--plan", phase.planFile,
    ]);
    status = {
      ...status,
      stage: "training",
      message: (
        `Training delay ${delayMinutes}m from ${phaseIndex === 0 ? "scratch" : "the previous phase"} `
        + `until validation base KL ≤ ${curriculum.targetValidationBaseKl} `
        + `or patience ${curriculum.patience}.`
      ),
    };
    writeStatus();
    await runStage("training", process.execPath, [
      path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
      path.join(repoRoot, "scripts/run-mlp-training.mjs"),
      "--plan", phase.planFile,
      "--training-only",
      "--skip-contract-build",
    ]);
    patchManifestCurriculum(phaseIndex, delayMinutes, phase);
    const completed = completedPhase(phaseIndex, delayMinutes, phase);
    summary.completedPhases = [
      ...summary.completedPhases.filter((entry) => entry.stage !== phaseIndex + 1),
      completed,
    ].sort((left, right) => left.stage - right.stage);
    writeSummary();
    status = {
      ...status,
      completedCurriculumPhases: summary.completedPhases.length,
      message: (
        `Completed delay ${delayMinutes}m; `
        + (phaseIndex + 1 < delaysMinutes.length
          ? `advancing to ${delaysMinutes[phaseIndex + 1]}m.`
          : "the delay curriculum is complete.")
      ),
    };
    writeStatus();
    pruneCompletedPhaseDataset(phaseIndex, phase);
    activePhaseFinalizeFile = undefined;
    if (fs.existsSync(finalizeFile)) {
      status = {
        ...status,
        stage: "paused",
        pausedAt: new Date().toISOString(),
        message: "Finalized the active phase and paused the remaining delay curriculum.",
      };
      writeStatus();
      process.exit(0);
    }
  }

  status = {
    ...status,
    stage: "complete",
    completedAt: new Date().toISOString(),
    completedCurriculumPhases: delaysMinutes.length,
    predictionDelayMs: 0,
    delayMinutes: 0,
    datasetDir: phasePaths(0).datasetDir,
    artifactDir: phasePaths(0).artifactDir,
    message: "Verified all 61 delay phases; the final 0-minute model is exported.",
  };
  writeStatus();
} catch (error) {
  if (interruptionSignal) {
    status = {
      ...status,
      stage: "paused",
      pausedAt: new Date().toISOString(),
      message: (
        `Paused by ${interruptionSignal}; completed datasets and phase checkpoints are resumable.`
      ),
    };
    writeStatus();
  } else {
    status = {
      ...status,
      stage: "failed",
      failedAt: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error),
    };
    writeStatus();
    throw error;
  }
} finally {
  clearInterval(finalizeForwarder);
}

function validateCurriculum(plan, source) {
  const value = plan.delayCurriculum;
  if (!plan.id || !plan.label || !plan.runDir || !plan.datasetDir || !plan.artifactDir) {
    throw new Error("Curriculum plan requires id, label, runDir, datasetDir, and artifactDir.");
  }
  if (plan.samplingIntervalMs !== 60_000) {
    throw new Error("Minute-candle delay curriculum requires samplingIntervalMs = 60000.");
  }
  if (!value
    || !Number.isInteger(value.startDelayMinutes)
    || !Number.isInteger(value.endDelayMinutes)
    || !Number.isInteger(value.stepMinutes)
    || value.stepMinutes <= 0
    || value.startDelayMinutes < value.endDelayMinutes
    || !Number.isInteger(value.epochsPerPhase)
    || value.epochsPerPhase <= 0
    || !Number.isInteger(value.patience)
    || value.patience <= 0
    || !Number.isFinite(value.targetValidationBaseKl)
    || value.targetValidationBaseKl <= 0
    || value.datasetRetention !== "anchor-and-active"
    || !Number.isFinite(value.minimumFreeSpaceGb)
    || value.minimumFreeSpaceGb <= 0) {
    throw new Error("Invalid delayCurriculum configuration.");
  }
  if (source.training?.targetRepresentation !== "minuteOracleProbabilities") {
    throw new Error("Delay curriculum requires persisted one-minute oracle targets.");
  }
  return value;
}

function delaySchedule(value) {
  const result = [];
  for (
    let delay = value.startDelayMinutes;
    delay >= value.endDelayMinutes;
    delay -= value.stepMinutes
  ) {
    result.push(delay);
  }
  if (result.at(-1) !== value.endDelayMinutes) {
    throw new Error("Delay step does not land exactly on endDelayMinutes.");
  }
  return result;
}

function phasePaths(delaySeconds) {
  const suffix = `delay-${delaySeconds}s`;
  const modelId = `${curriculumPlan.id}-${suffix}`;
  return {
    suffix,
    modelId,
    planFile: path.join(plansDir, `${suffix}.json`),
    datasetDir: path.join(
      repoRoot,
      "data",
      "ml-datasets",
      curriculumPlan.id,
      suffix,
    ),
    artifactDir: path.join(repoRoot, "data", "models", "mlp", modelId),
    runDir: path.join(phasesRunDir, suffix),
  };
}

function generatePhasePlan(phaseIndex, delayMinutes, phase, previousDelaySeconds) {
  const phaseZeroDataset = phasePaths(delaysMinutes[0] * 60).datasetDir;
  const componentSeeds = [
    ...(basePlan.componentSeedDatasetDirs ?? []),
    basePlan.datasetDir,
    ...(phaseIndex === 0 ? [] : [relative(phaseZeroDataset)]),
  ];
  const training = {
    ...basePlan.training,
    epochs: curriculum.epochsPerPhase,
    patience: curriculum.patience,
    targetValidation: {
      baseKlDivergence: curriculum.targetValidationBaseKl,
    },
    selectionMetric: "baseKlDivergence",
    featureStatisticsCache: relative(path.join(runDir, "feature-statistics.npz")),
    reuseFeatureStatisticsCache: phaseIndex > 0,
  };
  delete training.initializeFromCheckpoint;
  delete training.inheritedBestEpoch;
  if (previousDelaySeconds !== undefined) {
    training.initializeFromCheckpoint = relative(
      phasePaths(previousDelaySeconds).artifactDir + path.sep + "best-model.pt",
    );
  }
  return {
    ...basePlan,
    version: basePlan.version,
    id: phase.modelId,
    label: (
      `Direct 255-action MLP v12 · 1-minute candles/oracle · ${delayMinutes}m delay `
      + `· curriculum ${phaseIndex + 1}/${delaysMinutes.length}`
    ),
    planFile: relative(phase.planFile),
    datasetDir: relative(phase.datasetDir),
    componentSeedDatasetDirs: [...new Set(componentSeeds)],
    minuteOracleComponentSeedDatasetDirs: phaseIndex === 0
      ? []
      : [relative(phaseZeroDataset)],
    artifactDir: relative(phase.artifactDir),
    runDir: relative(phase.runDir),
    samplingIntervalMs: curriculumPlan.samplingIntervalMs,
    predictionDelayMs: delayMinutes * 60_000,
    teacherFit: {
      ...basePlan.teacherFit,
      adaptiveRounds: curriculum.skipTeacherRefinement
        ? 0
        : basePlan.teacherFit?.adaptiveRounds ?? 0,
    },
    training,
    curriculumTraining: {
      version: 1,
      stage: phaseIndex + 1,
      stages: delaysMinutes.length,
      delayMs: delayMinutes * 60_000,
      targetValidationMetric: "baseKlDivergence",
      targetValidationBaseKl: curriculum.targetValidationBaseKl,
      firstPhaseScratch: phaseIndex === 0,
    },
  };
}

async function runStage(stage, command, args) {
  status = {
    ...status,
    stage,
    stageStartedAt: new Date().toISOString(),
  };
  writeStatus();
  appendLog(`\n[${new Date().toISOString()}] ${stage}: ${command} ${args.join(" ")}\n`);
  const child = spawn(command, args, {
    cwd: repoRoot,
    env: environment,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  activeChild = child;
  consume(child.stdout, false);
  consume(child.stderr, true);
  const exitCode = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve(code ?? (signal ? 128 : 1)));
  });
  activeChild = undefined;
  if (interruptionSignal) throw new Error(`${stage} interrupted by ${interruptionSignal}.`);
  if (exitCode !== 0) throw new Error(`${stage} exited with code ${exitCode}.`);
}

function consume(stream, stderr) {
  const lines = readline.createInterface({ input: stream });
  lines.on("line", (line) => {
    appendLog(`${line}\n`);
    (stderr ? process.stderr : process.stdout).write(`${line}\n`);
    try {
      const event = JSON.parse(line);
      status = { ...status, latest: event };
      if (event.event === "epoch") {
        status = {
          ...status,
          epoch: event.epoch,
          epochs: event.epochs,
          globalStep: event.globalStep,
          train: event.train,
          validation: event.validation,
          bestValidation: event.bestValidation,
          bestEpoch: event.bestEpoch,
        };
      } else if (event.event === "train-step") {
        status = {
          ...status,
          epoch: event.epoch,
          epochs: event.epochs,
          globalStep: event.globalStep,
          latestStep: event,
        };
      } else if (event.event === "training-complete") {
        status = { ...status, finalMetrics: event };
      }
      writeStatus();
    } catch {
      // Human-readable verifier and build output remains in the append-only log.
    }
  });
}

function recoverCompletedPhase(phaseIndex, delayMinutes, phase) {
  const recorded = summary.completedPhases.find((entry) => entry.stage === phaseIndex + 1);
  if (recorded && phaseArtifactsComplete(phase)) return recorded;
  const phaseStatus = readJson(path.join(phase.runDir, "status.json"));
  if (phaseStatus?.stage === "complete" && phaseFilesComplete(phase)) {
    patchManifestCurriculum(phaseIndex, delayMinutes, phase);
    return completedPhase(phaseIndex, delayMinutes, phase);
  }
  return undefined;
}

function phaseArtifactsComplete(phase) {
  return [
    path.join(phase.artifactDir, "best-model.pt"),
    path.join(phase.artifactDir, "model.onnx"),
    path.join(phase.artifactDir, "manifest.json"),
  ].every((file) => fs.existsSync(file));
}

function phaseFilesComplete(phase) {
  return [
    path.join(phase.datasetDir, "dataset.json"),
    ...[
      "best-model.pt",
      "model.onnx",
      "manifest.json",
    ].map((file) => path.join(phase.artifactDir, file)),
  ].every((file) => fs.existsSync(file));
}

function pruneCompletedPhaseDataset(phaseIndex, phase) {
  if (curriculum.datasetRetention !== "anchor-and-active"
    || phaseIndex === 0
    || phaseIndex === delaysMinutes.length - 1) return;
  const datasetRoot = path.dirname(phasePaths(delaysMinutes[0] * 60).datasetDir);
  const target = path.resolve(phase.datasetDir);
  if (path.dirname(target) !== datasetRoot
    || !/^delay-\d+s$/.test(path.basename(target))) {
    throw new Error(`Refusing to prune an unexpected curriculum dataset path: ${target}`);
  }
  if (!fs.existsSync(target)) return;
  fs.rmSync(target, { recursive: true, force: true });
  appendLog(`${JSON.stringify({
    event: "curriculum-dataset-pruned",
    datasetDir: relative(target),
    retainedModels: true,
  })}\n`);
}

function ensureMinimumFreeSpace() {
  const datasetRoot = path.dirname(phasePaths(delaysMinutes[0] * 60).datasetDir);
  fs.mkdirSync(datasetRoot, { recursive: true });
  const stats = fs.statfsSync(datasetRoot);
  const freeBytes = Number(stats.bavail) * Number(stats.bsize);
  const requiredBytes = curriculum.minimumFreeSpaceGb * 1024 ** 3;
  if (freeBytes >= requiredBytes) return;
  throw new Error(
    `Only ${(freeBytes / 1024 ** 3).toFixed(2)} GiB is free; `
      + `${curriculum.minimumFreeSpaceGb} GiB is required before preparing the next phase.`,
  );
}

function completedPhase(phaseIndex, delayMinutes, phase) {
  const phaseStatus = readJson(path.join(phase.runDir, "status.json"));
  const manifest = readJsonRequired(path.join(phase.artifactDir, "manifest.json"));
  return {
    stage: phaseIndex + 1,
    delayMinutes,
    delayMs: delayMinutes * 60_000,
    modelId: phase.modelId,
    datasetDir: relative(phase.datasetDir),
    artifactDir: relative(phase.artifactDir),
    bestEpoch: manifest.training?.bestEpoch,
    bestValidation: manifest.training?.bestValidationMetrics,
    validationTargetReached: (
      phaseStatus?.finalMetrics?.validationTargetReached
      || manifest.training?.bestValidationMetrics?.baseKlDivergence
        <= curriculum.targetValidationBaseKl
    ),
    completedAt: phaseStatus?.completedAt ?? new Date().toISOString(),
  };
}

function patchManifestCurriculum(phaseIndex, delayMinutes, phase) {
  const manifestFile = path.join(phase.artifactDir, "manifest.json");
  const manifest = readJsonRequired(manifestFile);
  const lineage = delaysMinutes.slice(0, phaseIndex + 1).map((delay, index) => ({
    stage: index + 1,
    delayMs: delay * 60_000,
    weights: basePlan.training.lossWeights,
  }));
  manifest.training = {
    ...manifest.training,
    curriculum: {
      version: 1,
      stage: phaseIndex + 1,
      delayMs: delayMinutes * 60_000,
      parentKey: phaseIndex === 0
        ? ""
        : phasePaths(delaysMinutes[phaseIndex - 1] * 60).modelId,
      weightProfile: "fixed-ce-pmse-oracle-mi-half",
      lineage,
    },
  };
  atomicJson(manifestFile, manifest);
}

function clearPhaseMetrics() {
  const {
    epoch,
    epochs,
    globalStep,
    train,
    validation,
    bestValidation,
    bestEpoch,
    latestStep,
    finalMetrics,
    ...rest
  } = status;
  status = rest;
}

function writeSummary() {
  summary.updatedAt = new Date().toISOString();
  atomicJson(summaryFile, summary);
}

function writeStatus() {
  status.updatedAt = new Date().toISOString();
  atomicJson(statusFile, status);
}

function atomicJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(temporary, file);
}

function appendLog(value) {
  fs.appendFileSync(logFile, value);
}

function relative(file) {
  return path.relative(repoRoot, file).split(path.sep).join("/");
}

function readJsonRequired(file) {
  const value = readJson(file);
  if (value === undefined) throw new Error(`Missing JSON file: ${file}`);
  return value;
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return undefined;
    throw error;
  }
}

function processIsAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function argument(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
