import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const planArgument = argument("plan") ?? "ml/training-plan.json";
const trainingOnly = process.argv.includes("--training-only");
const datasetOnly = process.argv.includes("--dataset-only");
const verificationOnly = process.argv.includes("--verification-only");
const frozenStudyOnly = process.argv.includes("--frozen-study-only");
const productionMinuteOnly = process.argv.includes("--production-minute");
const derivedTrainingOnly = frozenStudyOnly || productionMinuteOnly;
if ([trainingOnly, datasetOnly, verificationOnly, frozenStudyOnly, productionMinuteOnly]
  .filter(Boolean).length > 1) {
  throw new Error(
    "Choose only one of --training-only, --dataset-only, --verification-only, "
    + "--frozen-study-only, or --production-minute.",
  );
}
const planFile = path.resolve(repoRoot, planArgument);
const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
const runDir = path.resolve(repoRoot, plan.runDir);
const datasetDir = path.resolve(
  repoRoot,
  productionMinuteOnly
    ? plan.productionTraining?.outputDatasetDir ?? plan.datasetDir
    : frozenStudyOnly
    ? plan.frozenStudySampling?.outputDatasetDir ?? plan.datasetDir
    : plan.datasetDir,
);
const artifactDir = path.resolve(repoRoot, plan.artifactDir);
let trainingPlan = plan;
let trainingPlanFile = planFile;
let trainingDatasetDir = datasetDir;
let trainingArtifactDir = artifactDir;
const statusFile = path.join(runDir, "status.json");
const logFile = path.join(runDir, "training.log");
const finalizeFile = path.join(runDir, "FINALIZE");
const runtimeRoot = path.join(repoRoot, "data", "runtime-cache");
const temporaryDirectory = path.join(runtimeRoot, "tmp");
const tritonCacheDirectory = path.join(runtimeRoot, "triton");
const torchInductorCacheDirectory = path.join(runtimeRoot, "torchinductor");
const cudaCacheDirectory = path.join(runtimeRoot, "cuda");
fs.mkdirSync(runDir, { recursive: true });
for (const directory of [
  temporaryDirectory,
  tritonCacheDirectory,
  torchInductorCacheDirectory,
  cudaCacheDirectory,
]) {
  fs.mkdirSync(directory, { recursive: true });
}

const previous = readJson(statusFile);
if (previous?.pid && processIsAlive(previous.pid)
  && !["complete", "failed", "paused"].includes(previous.stage)) {
  throw new Error(`MLP training is already running as PID ${previous.pid}.`);
}
fs.rmSync(finalizeFile, { force: true });

let activeChild;
let interruptionSignal;
let status = {
  planId: plan.id,
  pid: process.pid,
  stage: "starting",
  startedAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  datasetDir,
  artifactDir,
  logFile,
};
writeStatus();

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    interruptionSignal ??= signal;
    if (activeChild && !activeChild.killed) activeChild.kill(signal);
  });
}

try {
  await runStage("feature-contract-build", process.execPath, [
    path.join(repoRoot, "node_modules/typescript/bin/tsc"),
    "-p", path.join(repoRoot, "packages/bot-algo/tsconfig.json"),
  ]);
  if (verificationOnly) {
    await runStage("verification", process.execPath, [
      path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
      path.join(repoRoot, "scripts/verify-mlp-model.mjs"),
      artifactDir,
    ]);
    status = {
      ...status,
      stage: "complete",
      completedAt: new Date().toISOString(),
      message: "Verified model artifact is available to the backend and UI.",
    };
    writeStatus();
    process.exit(0);
  }
  if (!trainingOnly) {
    await runStage("cuda-build", process.execPath, [
      path.join(repoRoot, "scripts/build-vw-kama-cuda.mjs"),
    ]);
    if (derivedTrainingOnly) {
      if (productionMinuteOnly && !plan.productionTraining) {
        throw new Error(
          "--production-minute requires productionTraining in the plan.",
        );
      }
      if (frozenStudyOnly && !plan.frozenStudySampling) {
        throw new Error(
          "--frozen-study-only requires frozenStudySampling in the plan.",
        );
      }
      await runStage(
        productionMinuteOnly ? "production-training-source" : "frozen-study-source",
        process.execPath,
        [
        path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
        path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
        "--plan", planFile,
        "--preparation-mode", "frozen-study-source",
        ],
      );
      const frozenPlanFile = path.resolve(
        repoRoot,
        productionMinuteOnly
          ? plan.productionTraining.outputPlanFile
          : plan.frozenStudySampling.outputPlanFile,
      );
      await runStage(
        productionMinuteOnly
          ? "production-training-materialization"
          : "frozen-study-materialization",
        process.execPath,
        [
        path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
        path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
        "--plan", frozenPlanFile,
        ],
      );
      trainingPlanFile = frozenPlanFile;
      trainingPlan = JSON.parse(fs.readFileSync(trainingPlanFile, "utf8"));
      trainingDatasetDir = path.resolve(repoRoot, trainingPlan.datasetDir);
      trainingArtifactDir = path.resolve(repoRoot, trainingPlan.artifactDir);
      status = {
        ...status,
        planId: trainingPlan.id,
        datasetDir: trainingDatasetDir,
        artifactDir: trainingArtifactDir,
        message: productionMinuteOnly
          ? "Full production corpus is complete; starting one-minute-oracle training."
          : "Frozen production-wide sample is complete; starting one-minute-oracle training.",
      };
      writeStatus();
    } else {
      await runStage("dataset", process.execPath, [
        path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
        path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
        "--plan", planFile,
      ]);
    }
    if (!derivedTrainingOnly && (plan.teacherFit?.adaptiveRounds ?? 0) > 0) {
      await runStage("dataset-refinement", process.execPath, [
        path.join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
        path.join(repoRoot, "scripts/build-mlp-dataset.ts"),
        "--plan", planFile,
        "--refinement-pass", "1",
      ]);
    }
  }
  if (datasetOnly) {
    status = {
      ...status,
      stage: "complete",
      completedAt: new Date().toISOString(),
      message: "Reusable input/oracle components and the delayed dataset pairing are complete.",
    };
    writeStatus();
    process.exit(0);
  }
  const training = trainingPlan.training;
  const python = path.join(
    repoRoot,
    ".venv-ml",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
  if (!fs.existsSync(python)) {
    throw new Error("ML environment is missing. Run `npm run mlp:bootstrap` first.");
  }
  await runStage("training", python, [
    path.join(repoRoot, "ml/train_mlp.py"),
    "--dataset", trainingDatasetDir,
    "--output", trainingArtifactDir,
    "--model-id", trainingPlan.id,
    "--label", trainingPlan.label,
    "--plan", trainingPlanFile,
    "--target", training.targetRepresentation ?? "rawOracleProbabilities",
    "--epochs", String(training.epochs),
    "--batch-size", String(training.batchSize),
    "--evaluation-batch-size", String(training.evaluationBatchSize ?? training.batchSize),
    "--validation-fraction", String(training.validationFraction ?? 1),
    "--training-fraction", String(training.trainingFraction ?? 1),
    "--accumulate", String(training.gradientAccumulation),
    "--learning-rate", String(training.learningRate),
    "--weight-decay", String(training.weightDecay),
    "--dropout", String(training.dropout),
    "--states-per-example", String(training.statesPerExample),
    "--patience", String(training.patience),
    "--workers", String(training.workers),
    "--seed", String(training.seed),
    "--device", training.device,
    "--log-every-steps", String(training.logEverySteps),
    "--selection-metric", training.selectionMetric ?? "loss",
    "--loss-weights-json", JSON.stringify(training.lossWeights),
    "--time-weighting-json", JSON.stringify(training.timeWeighting),
    "--feature-statistics-cache", path.join(runDir, "feature-statistics.npz"),
    "--finalize-file", finalizeFile,
    "--resume",
    ...(training.initializeFromCheckpoint
      ? [
          "--initialize-from-checkpoint",
          path.resolve(repoRoot, training.initializeFromCheckpoint),
        ]
      : []),
    ...(Number.isInteger(training.inheritedBestEpoch)
      ? ["--inherited-best-epoch", String(training.inheritedBestEpoch)]
      : []),
    ...(training.evaluationOnly ? ["--evaluation-only"] : []),
    ...(Number.isFinite(training.targetValidation?.klDivergence)
      ? ["--target-validation-kl", String(training.targetValidation.klDivergence)]
      : []),
    ...(Number.isFinite(training.targetValidation?.klDivergenceStdDev)
      ? ["--target-validation-kl-stddev", String(training.targetValidation.klDivergenceStdDev)]
      : []),
    ...(training.weightedTrainingSample ? ["--weighted-training-sample"] : []),
    ...(training.compile ? ["--compile"] : []),
  ]);
  await runStage("verification", process.execPath, [
    path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
    path.join(repoRoot, "scripts/verify-mlp-model.mjs"),
    trainingArtifactDir,
  ]);
  status = {
    ...status,
    stage: "complete",
    completedAt: new Date().toISOString(),
    message: derivedTrainingOnly
      ? "Verified one-minute-oracle model is available to the backend and UI."
      : "Verified model artifact is available to the backend and UI.",
  };
  writeStatus();
} catch (error) {
  if (interruptionSignal) {
    status = {
      ...status,
      stage: "paused",
      pausedAt: new Date().toISOString(),
      message: `Paused by ${interruptionSignal}; completed shards and checkpoints are resumable.`,
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
}

async function runStage(stage, command, args) {
  status = { ...status, stage, stageStartedAt: new Date().toISOString() };
  writeStatus();
  appendLog(`\n[${new Date().toISOString()}] ${stage}: ${command} ${args.join(" ")}\n`);
  const child = spawn(command, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONUTF8: process.env.PYTHONUTF8 || "1",
      PYTHONIOENCODING: process.env.PYTHONIOENCODING || "utf-8",
      TMPDIR: process.env.TRADING_ML_TMP_DIR || temporaryDirectory,
      TEMP: process.env.TRADING_ML_TMP_DIR || temporaryDirectory,
      TMP: process.env.TRADING_ML_TMP_DIR || temporaryDirectory,
      TRITON_CACHE_DIR: process.env.TRITON_CACHE_DIR || tritonCacheDirectory,
      TORCHINDUCTOR_CACHE_DIR:
        process.env.TORCHINDUCTOR_CACHE_DIR || torchInductorCacheDirectory,
      CUDA_CACHE_PATH: process.env.CUDA_CACHE_PATH || cudaCacheDirectory,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  activeChild = child;
  consume(child.stdout, false);
  consume(child.stderr, true);
  const exitCode = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve(code ?? (signal ? 128 : 1)));
  });
  activeChild = undefined;
  if (interruptionSignal) {
    throw new Error(`${stage} interrupted by ${interruptionSignal}.`);
  }
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
      } else if (event.event === "dataset-source-rejected") {
        status = {
          ...status,
          sourceRecovery: status.sourceRecovery
            ? { ...status.sourceRecovery, active: false, failedAt: new Date().toISOString() }
            : undefined,
          sourceRejections: {
            count: event.rejectedDays,
            latestDate: event.date,
            latestDetail: event.detail,
            queue: path.join(datasetDir, "source-rejection-queue.json"),
          },
        };
      } else if (event.event === "dataset-source-recovered") {
        status = {
          ...status,
          sourceRejections: {
            ...status.sourceRejections,
            count: event.remainingRejectedDays,
          },
        };
      } else if (event.event === "dataset-source-recovery-start") {
        status = {
          ...status,
          sourceRecovery: {
            active: true,
            date: event.date,
            detail: event.detail,
            startedAt: new Date().toISOString(),
          },
        };
      } else if (event.event === "dataset-source-recovery-complete") {
        status = {
          ...status,
          sourceRecovery: {
            active: false,
            date: event.date,
            candles: event.candles,
            elapsedMs: event.elapsedMs,
            completedAt: new Date().toISOString(),
          },
        };
      }
      writeStatus();
    } catch {
      // Human-readable verifier output is still retained in the log.
    }
  });
}

function writeStatus() {
  status.updatedAt = new Date().toISOString();
  const temporary = `${statusFile}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(status, null, 2)}\n`);
  fs.renameSync(temporary, statusFile);
}

function appendLog(value) {
  fs.appendFileSync(logFile, value);
}

function processIsAlive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
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

function argument(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
