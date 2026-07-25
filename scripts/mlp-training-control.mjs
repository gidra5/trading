import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const action = process.argv[2] ?? "status";
const planIndex = process.argv.indexOf("--plan");
const planFile = path.resolve(repoRoot, planIndex >= 0 ? process.argv[planIndex + 1] : "ml/training-plan.json");
const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
const runDir = path.resolve(repoRoot, plan.runDir);
const statusFile = path.join(runDir, "status.json");
const finalizeFile = path.join(runDir, "FINALIZE");

if (action === "finalize") {
  fs.mkdirSync(runDir, { recursive: true });
  fs.writeFileSync(finalizeFile, `${new Date().toISOString()}\n`);
  process.stdout.write(
    "Finalize requested. Training will validate the current update, export the best checkpoint, "
    + "run CPU/GPU parity verification, and publish it to the UI model list.\n",
  );
} else if (action === "status") {
  const follow = !process.argv.includes("--once");
  showStatus();
  if (follow) {
    process.stdout.write("Watching training status; press Ctrl+C to stop watching (training continues).\n");
    setInterval(showStatus, 2_000);
  }
} else {
  throw new Error("Usage: node scripts/mlp-training-control.mjs status [--once] | finalize");
}

function showStatus() {
  let status;
  try {
    status = JSON.parse(fs.readFileSync(statusFile, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") {
      process.stdout.write("No training run has started yet.\n");
      return;
    }
    throw error;
  }
  process.stdout.write(`\x1b[2J\x1b[H${format(status)}\n`);
}

function format(status) {
  const lines = [
    `${status.planId} · ${status.stage} · updated ${status.updatedAt}`,
  ];
  const datasetStage = typeof status.stage === "string"
    && (status.stage.startsWith("dataset") || status.stage.startsWith("frozen-study"));
  const datasetProgress = status.latest?.event === "dataset-progress"
    ? status.latest
    : datasetStage
      ? latestLoggedEvent(status.logFile, "dataset-progress")
      : undefined;
  const oracleProgress = status.latest?.event === "dataset-oracle"
    ? status.latest
    : datasetStage
      ? latestLoggedEvent(status.logFile, "dataset-oracle")
      : undefined;
  if (oracleProgress) {
    lines.push(
      `Oracle ${oracleProgress.backend?.toUpperCase() ?? "—"}`
      + ` ${oracleProgress.day}/${oracleProgress.days} days · ${oracleProgress.date}`,
      `Oracle kernel ${number(oracleProgress.kernelMs)} ms`
      + ` · wall ${number(oracleProgress.wallMs)} ms`
      + ` · ${oracleProgress.candles ?? "—"} candles`,
    );
  }
  if (datasetProgress) {
    const value = datasetProgress;
    lines.push(
      `Dataset ${value.day}/${value.days} days · ${value.date} ${value.split ?? value.component ?? ""}`.trim(),
      `Teacher fits ${value.examplesCompleted}/${value.examplesTotal}`,
    );
    if (Number.isFinite(value.examplesPerSecond)) {
      lines.push(
        `Teacher ${number(value.examplesPerSecond)} fits/s · GPU ${number(value.gpuMemoryMiB)} MiB`
        + ` · KL ${number(value.meanKlDivergence)} · pMSE ${number(value.meanSquaredError)}`,
      );
    }
    if (Number.isFinite(value.temporalWarmSelectedFraction)) {
      lines.push(
        `Temporal warm selections ${(100 * value.temporalWarmSelectedFraction).toFixed(1)}%`
        + ` · normalized step ${number(value.temporalMeanNormalizedStepBefore)}`
        + ` → ${number(value.temporalMeanNormalizedStepAfter)}`,
      );
    }
  }
  const featureProgress = status.latest?.event === "dataset-feature-refresh-progress"
    ? status.latest
    : status.stage === "dataset-features"
      ? latestLoggedEvent(status.logFile, "dataset-feature-refresh-progress")
      : undefined;
  if (featureProgress) {
    lines.push(
      `Feature refresh ${featureProgress.completedShards}/${featureProgress.shards} shards`
      + ` · day ${featureProgress.day}/${featureProgress.days}`
      + ` · ${featureProgress.date} ${featureProgress.split}`,
      `Latest refreshed shard ${featureProgress.examples} examples`,
    );
  }
  if (status.stage === "dataset-refinement" && status.datasetDir) {
    const progress = readOptionalJson(path.join(status.datasetDir, "progress.json"));
    const queue = readOptionalJson(path.join(status.datasetDir, "teacher-refinement-queue.json"));
    const refinedShards = Array.isArray(progress?.shards)
      ? progress.shards.filter((shard) => (shard.refinementPass ?? 0) > 0).length
      : undefined;
    const remainingFits = Array.isArray(queue?.cases) ? queue.cases.length : undefined;
    if (refinedShards !== undefined || remainingFits !== undefined) {
      lines.push(
        `Refined shards ${refinedShards ?? "—"}`
        + ` · remaining queued teacher fits ${remainingFits ?? "—"}`,
      );
    }
  }
  if (status.sourceRejections?.count > 0) {
    lines.push(
      `Unrecovered source days ${status.sourceRejections.count}`
      + ` · latest ${status.sourceRejections.latestDate ?? "—"}`,
      `Source refinement queue: ${status.sourceRejections.queue}`,
    );
  }
  if (status.sourceRecovery?.active) {
    lines.push(
      `Recovering source day ${status.sourceRecovery.date ?? "—"}`,
      status.sourceRecovery.detail ?? "Downloading and validating the complete daily shard",
    );
  }
  const step = status.latestStep;
  if (step) {
    lines.push(
      `Epoch ${step.epoch + 1}/${step.epochs} · batch ${step.batch}/${step.batches} · update ${step.globalStep}`,
      `LR ${number(step.learningRate)} · grad ${number(step.gradientNorm)} · ${step.examplesPerSecond} examples/s · GPU ${step.gpuMemoryMiB} MiB`,
      metricLine("Latest", step.latest),
    );
  }
  if (status.train) lines.push(metricLine("Train", status.train));
  if (status.validation) lines.push(metricLine("Validation", status.validation));
  if (Number.isFinite(status.bestValidation)) {
    lines.push(`Best validation loss ${number(status.bestValidation)} at epoch ${(status.bestEpoch ?? -1) + 1}`);
  }
  if (status.finalMetrics) {
    lines.push(metricLine("Final test", status.finalMetrics.test));
  }
  if (status.message) lines.push(status.message);
  if (status.error) lines.push(`ERROR: ${status.error}`);
  lines.push(`Log: ${status.logFile}`);
  return lines.join("\n");
}

function readOptionalJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return undefined;
    throw error;
  }
}

function metricLine(label, metrics) {
  return `${label}: loss ${number(metrics.loss)} · KL ${number(metrics.klDivergence)}`
    + ` ± ${number(metrics.klDivergenceStdDev)}`
    + ` (var ${number(metrics.klDivergenceVariance)})`
    + ` · deployment KL ${number(metrics.deploymentKlDivergence)}`
    + ` ± ${number(metrics.deploymentKlDivergenceStdDev)}`
    + ` · pMSE ${number(metrics.probabilityMse)}`
    + ` ± ${number(metrics.probabilityMseStdDev)}`
    + ` (var ${number(metrics.probabilityMseVariance)})`
    + ` · paramMSE ${number(metrics.parameterMse)}`
    + ` · excess H ${number(metrics.excessEntropy)} · Temporal MI ${number(metrics.temporalMutualInformation)}`
    + ` · Temporal reward ${number(metrics.temporalMutualInformationReward)}`
    + ` · Oracle MI ${number(metrics.oracleMutualInformation)}`;
}

function number(value) {
  return Number.isFinite(value) ? Number(value).toPrecision(6) : "—";
}

function latestLoggedEvent(file, eventName) {
  try {
    const descriptor = fs.openSync(file, "r");
    try {
      const size = fs.fstatSync(descriptor).size;
      const length = Math.min(size, 64 * 1024);
      const buffer = Buffer.allocUnsafe(length);
      fs.readSync(descriptor, buffer, 0, length, size - length);
      const lines = buffer.toString("utf8").split("\n");
      for (let index = lines.length - 1; index >= 0; index -= 1) {
        try {
          const value = JSON.parse(lines[index]);
          if (value?.event === eventName) return value;
        } catch {
          // The tail can begin within a line and also contains human-readable output.
        }
      }
    } finally {
      fs.closeSync(descriptor);
    }
  } catch {
    // The status file remains useful if its append-only log is unavailable.
  }
  return undefined;
}
