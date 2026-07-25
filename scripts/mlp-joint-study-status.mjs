import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const planIndex = process.argv.indexOf("--plan");
const planFile = path.resolve(repoRoot, planIndex >= 0
  ? process.argv[planIndex + 1]
  : "ml/training-plan.json");
const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
const studyIndex = process.argv.indexOf("--study");
const studyKey = studyIndex >= 0 ? process.argv[studyIndex + 1] : "lossWeightStudy";
if (!plan[studyKey]?.runDir) throw new Error(`Unknown study configuration '${studyKey}'.`);
const statusFile = path.resolve(repoRoot, plan[studyKey].runDir, "status.json");
const queueFile = path.resolve(repoRoot, plan[studyKey].runDir, "queue.json");
const follow = !process.argv.includes("--once");

show();
if (follow) {
  process.stdout.write("Watching joint delay/weight study; Ctrl+C stops only this watcher.\n");
  setInterval(show, 2_000);
}

function show() {
  let status;
  try {
    status = JSON.parse(fs.readFileSync(statusFile, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") {
      const queue = readJson(queueFile);
      process.stdout.write(queue
        ? `Exhaustive study queue · ${queue.stage} · updated ${queue.updatedAt}`
          + `${queue.sourceCompletedRuns != null
            ? ` · fractional ${queue.sourceCompletedRuns}/${queue.sourcePlannedRuns}`
            : ""}\n`
        : "Joint delay/weight study has not started.\n");
      return;
    }
    throw error;
  }
  const lines = [
    `${status.planId} · ${status.design ?? "joint"} delay/weight study`
      + ` · ${status.stage} · updated ${status.updatedAt}`,
    `PID ${status.pid} · ${status.completedRuns ?? 0}/${status.variants ?? "—"} combinations complete`,
  ];
  if (status.supervisorHeartbeatAt) {
    const heartbeatAge = ageSeconds(status.supervisorHeartbeatAt);
    lines.push(
      `Supervisor heartbeat ${heartbeatAge}s ago`
      + `${heartbeatAge > 15 ? " · STALE" : " · healthy"}`,
    );
  }
  if (status.child?.pid) {
    lines.push(
      `Child PID ${status.child.pid} · attempt ${status.child.attempt}/${status.child.maxAttempts}`
      + ` · no progress ${status.child.noProgressSeconds ?? 0}s`
      + `${status.child.watchdogTimeoutMs
        ? `/${Math.floor(status.child.watchdogTimeoutMs / 1_000)}s watchdog`
        : ""}`,
    );
  }
  if (status.sleepInhibitor) {
    lines.push(
      status.sleepInhibitor.active
        ? `Windows standby inhibition active`
          + `${status.sleepInhibitor.windowsPid
            ? ` · Windows PID ${status.sleepInhibitor.windowsPid}`
            : ""}`
        : `Windows standby inhibition INACTIVE`
          + `${status.sleepInhibitor.error ? ` · ${status.sleepInhibitor.error}` : ""}`,
    );
  }
  if (status.storage) {
    const linux = status.storage.linux;
    const windows = status.storage.windowsHost;
    lines.push(
      `Disk free: Linux ${gib(linux)}`
      + `${windows ? ` · Windows C: ${gib(windows)}` : ""}`,
    );
  }
  if (status.stage === "priority-dataset") {
    const days = status.selectedDays;
    const examples = status.selectedExamples;
    if (days) lines.push(`Priority subset days: ${days.train} train / ${days.validation} validation / ${days.test} test`);
    if (examples) lines.push(`Priority examples: ${examples.train} train / ${examples.validation} validation / ${examples.test} test`);
    const latest = status.latest;
    if (latest?.day && latest?.days) lines.push(`Priority fit day ${latest.day}/${latest.days} · ${latest.date}`);
  } else if (status.stage === "waiting-for-components") {
    lines.push(`Base preparation: ${status.basePreparation?.stage ?? "waiting"}`);
    const latest = status.basePreparation?.latest;
    if (latest?.day && latest?.days) lines.push(`Component day ${latest.day}/${latest.days} · ${latest.date}`);
  }
  if (Number.isFinite(status.delayMinutes)) {
    lines.push(
      `Current delay ${status.delayMinutes}m · delay ${status.delayIndex}/${status.delays}`,
      `Weight setting ${status.weightVariantKey ?? "pairing"}`
        + `${status.weightVariant ? ` · ${status.weightVariant}/${status.weightVariants}` : ""}`
        + `${status.variant ? ` · overall ${status.variant}/${status.variants}` : ""}`,
    );
  }
  if (Number.isFinite(status.epoch)) {
    lines.push(`Epoch ${status.epoch + 1}/${status.epochs} · best validation KL ${number(status.bestValidation)} at epoch ${(status.bestEpoch ?? -1) + 1}`);
  }
  if (Number.isFinite(status.populationEpoch)) {
    lines.push(
      `Population ${status.latest?.populationGroup ?? "—"}/${status.latest?.populationGroups ?? "—"}`
      + ` · epoch ${status.populationEpoch + 1}/${status.populationEpochs}`
      + ` · width ${status.effectiveTrainingPopulationSize
        ?? status.trainingPopulationSize ?? "—"}`,
    );
  }
  if (Number.isFinite(status.completedPopulationJobs)
    && Number.isFinite(status.pendingPopulationJobs)) {
    lines.push(
      `Population candidates ${status.completedPopulationJobs}/${status.pendingPopulationJobs}`
      + " persisted for this delay",
    );
  }
  if (status.latest?.event === "train-step") {
    lines.push(
      `Live epoch ${status.latest.epoch + 1}/${status.latest.epochs}`
      + ` · batch ${status.latest.batch}/${status.latest.batches}`
      + ` · ${number(status.latest.examplesPerSecond)} examples/s`
      + ` · GPU allocation ${number(status.latest.gpuMemoryMiB)} MiB`
      + `${status.latest.job ? ` · ${status.latest.job}` : ""}`,
    );
  }
  if (status.latest?.event === "population-train-step") {
    lines.push(
      `Live population ${status.latest.populationGroup}/${status.latest.populationGroups}`
      + ` · epoch ${status.latest.epoch + 1}/${status.latest.epochs}`
      + ` · batch ${status.latest.batch}/${status.latest.batches}`
      + ` · ${number(status.latest.populationExamplesPerSecond)} effective examples/s`
      + ` · GPU allocation ${number(status.latest.gpuMemoryMiB)} MiB`,
      `Members: ${(status.latest.jobs ?? []).join(", ")}`,
    );
  }
  if (Number.isFinite(status.bestValidationKl)) {
    lines.push(
      `Best observed ${status.bestDelayMinutes}m / ${status.bestWeightVariant}`
      + ` · validation KL ${number(status.bestValidationKl)}`,
    );
  }
  if (status.message) lines.push(status.message);
  if (status.lastWatchdog) {
    lines.push(
      `Last watchdog: ${status.lastWatchdog.stage} attempt ${status.lastWatchdog.attempt}`
      + ` after ${status.lastWatchdog.noProgressSeconds}s`,
      `Diagnostics: ${status.lastWatchdog.diagnosticFile}`,
    );
  }
  if (status.error) lines.push(`ERROR: ${status.error}`);
  lines.push(`Log: ${status.logFile}`);
  process.stdout.write(`\x1b[2J\x1b[H${lines.join("\n")}\n`);
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT" || error instanceof SyntaxError) return undefined;
    throw error;
  }
}

function number(value) {
  return Number.isFinite(value) ? Number(value).toPrecision(7) : "—";
}

function ageSeconds(timestamp) {
  const milliseconds = Date.now() - Date.parse(timestamp);
  return Number.isFinite(milliseconds) ? Math.max(0, Math.floor(milliseconds / 1_000)) : Infinity;
}

function gib(storage) {
  if (!storage || !Number.isFinite(storage.freeGiB)) return "unknown";
  return `${storage.freeGiB.toFixed(1)} GiB${storage.healthy ? "" : " · LOW"}`;
}
