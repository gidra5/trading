import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const plan = JSON.parse(
  fs.readFileSync(path.join(repoRoot, "ml/training-plan.json"), "utf8"),
);
const studyKeyIndex = process.argv.indexOf("--study-key");
const studyKey = studyKeyIndex >= 0
  ? process.argv[studyKeyIndex + 1]
  : "adaptiveCurriculumStudy";
if (!studyKey || !plan[studyKey]) {
  throw new Error(`Unknown adaptive curriculum study: ${studyKey ?? "—"}`);
}
const configured = plan[studyKey];
const configuration = configured.extends
  ? { ...plan[configured.extends], ...configured }
  : configured;
const runDir = path.resolve(repoRoot, configuration.runDir);
const outputDir = path.resolve(repoRoot, configuration.outputDir);
const statusFile = path.join(runDir, "status.json");
const summaryFile = path.join(outputDir, "summary.json");
const logFile = path.join(runDir, "study.log");
const watch = process.argv.includes("--watch");

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return undefined;
    throw error;
  }
}

function alive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function number(value) {
  return Number.isFinite(value) ? Number(value).toPrecision(6) : "—";
}

function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function recentEvents(eventNames, maximumBytes = 2 * 1024 * 1024) {
  let descriptor;
  try {
    const size = fs.statSync(logFile).size;
    const length = Math.min(size, maximumBytes);
    const buffer = Buffer.alloc(length);
    descriptor = fs.openSync(logFile, "r");
    fs.readSync(descriptor, buffer, 0, length, size - length);
    const lines = buffer.toString("utf8").split("\n");
    const found = new Map();
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      let event;
      try {
        event = JSON.parse(lines[index]);
      } catch {
        continue;
      }
      if (eventNames.has(event?.event) && !found.has(event.event)) {
        found.set(event.event, event);
        if (found.size === eventNames.size) break;
      }
    }
    return found;
  } catch (error) {
    if (error?.code === "ENOENT") return new Map();
    throw error;
  } finally {
    if (descriptor !== undefined) fs.closeSync(descriptor);
  }
}

function jobKeys(event) {
  return (event?.jobs ?? []).map((job) =>
    typeof job === "string" ? job : job?.key).filter(Boolean);
}

function samePopulation(left, right) {
  if (!left || !right) return false;
  const leftKeys = new Set(jobKeys(left));
  return Number(left.populationGroup) === Number(right.populationGroup)
    && jobKeys(right).some((key) => leftKeys.has(key));
}

function percentage(value, total) {
  if (!Number.isFinite(value) || !Number.isFinite(total) || total <= 0) {
    return "—";
  }
  return `${(100 * value / total).toFixed(1)}%`;
}

function render() {
  const status = readJson(statusFile);
  const phaseSummary = readJson(summaryFile);
  const bootstrapSummary = configuration.bootstrapSummary
    ? readJson(path.resolve(repoRoot, configuration.bootstrapSummary))
    : undefined;
  const summary = phaseSummary ?? bootstrapSummary;
  const state = !phaseSummary
      && bootstrapSummary?.state
      && Number.isInteger(configuration.bootstrapStepSeconds)
    ? {
        ...bootstrapSummary.state,
        trial_delay_seconds: Math.max(
          configuration.minimumDelaySeconds,
          bootstrapSummary.state.anchor_delay_seconds
            - configuration.bootstrapStepSeconds,
        ),
        step_seconds: configuration.bootstrapStepSeconds,
        dwell_epochs: 0,
        stale_epochs: 0,
        best_validation: null,
      }
    : summary?.state ?? {
        anchor_delay_seconds: configuration.initialDelaySeconds,
        trial_delay_seconds: Math.max(
          configuration.minimumDelaySeconds,
          configuration.initialDelaySeconds - configuration.initialStepSeconds,
        ),
        step_seconds: configuration.initialStepSeconds,
      };
  const best = summary?.bestObserved?.validation;
  const lines = [
    `${plan.id} · ${studyKey} · adaptive absolute-weight/delay curriculum`,
    status
      ? `${status.stage ?? "unknown"} · updated ${status.updatedAt ?? "—"}`
      : "not started",
    `PID ${status?.pid ?? "—"} · ${alive(status?.pid) ? "running" : "not running"}`,
    `Phase rounds ${phaseSummary?.completedRounds ?? status?.completedRounds ?? 0}`
      + `/${configuration.maximumRounds}`,
    `Delay anchor ${duration(state?.anchor_delay_seconds)}`
      + ` · trial ${duration(status?.delaySeconds ?? state?.trial_delay_seconds)}`
      + ` · step ${duration(state?.step_seconds)}`,
  ];
  if (status?.fidelity) {
    lines.push(
      `Fidelity ${status.fidelity}`
      + (status.fidelityPhase ? `/${status.fidelityPhase}` : "")
      + (Number.isFinite(status.branches) ? ` · ${status.branches} branches` : ""),
    );
  }
  if (Number.isFinite(status?.bestValidationKl)) {
    lines.push(
      `Last best/reference KL ${number(status.bestValidationKl)}`
      + ` / ${number(status.referenceValidationKl)}`
      + (status.decision ? ` · ${status.decision}` : ""),
    );
  }
  if (best) {
    lines.push(
      `Best observed validation KL ${number(best.klDivergence)}`
      + ` ± ${number(best.klDivergenceStdDev)}`,
    );
    if (Number.isFinite(best.deploymentKlDivergence)) {
      lines.push(
        `Best observed deployment KL ${number(best.deploymentKlDivergence)}`
        + ` ± ${number(best.deploymentKlDivergenceStdDev)}`,
      );
    }
  }
  const schedule = phaseSummary?.scheduleOptimization?.recommended;
  if (schedule) {
    lines.push(
      `Recommended schedule: ${duration(schedule.progressSeconds)} advanced`
      + ` · ${schedule.trainingBatches} batches`
      + ` · KL gap ${number(schedule.qualityGap)}`
      + ` · frontier ${phaseSummary.scheduleOptimization.paretoFront.length}`,
    );
  }
  const populationEvents = recentEvents(new Set([
    "population-training-start",
    "population-group-start",
    "population-train-step",
    "population-epoch",
    "population-group-complete",
  ]));
  const latest = status?.latest;
  const populationStage = status?.stage?.startsWith("adaptive-");
  const liveStep = populationStage && latest?.event === "population-train-step"
    ? latest
    : populationStage
      ? populationEvents.get("population-train-step")
      : undefined;
  const liveEpoch = populationStage && latest?.event === "population-epoch"
    ? latest
    : populationStage
      ? populationEvents.get("population-epoch")
      : undefined;
  const groupStart = populationStage
    ? populationEvents.get("population-group-start")
    : undefined;
  const trainingStart = populationStage
    ? populationEvents.get("population-training-start")
    : undefined;
  if (liveStep && alive(status?.pid)) {
    const populationSize = Math.max(1, jobKeys(liveStep).length);
    const batchSize = Number(plan.training?.batchSize);
    const rate = Number(liveStep.populationExamplesPerSecond);
    const remainingExamples = (
      Math.max(0, Number(liveStep.batches) - Number(liveStep.batch))
      * batchSize
      * populationSize
    );
    const epochEta = rate > 0 ? remainingExamples / rate : Number.NaN;
    lines.push(
      `Population group ${liveStep.populationGroup}/${liveStep.populationGroups}`
      + ` · epoch ${liveStep.epoch + 1}/${liveStep.epochs}`
      + ` · batch ${liveStep.batch}/${liveStep.batches}`
      + ` (${percentage(liveStep.batch, liveStep.batches)})`
      + ` · epoch ETA ${duration(epochEta)}`,
      `Throughput ${number(rate)} effective examples/s`
      + ` · GPU ${number(liveStep.gpuMemoryMiB)} MiB`
      + ` · LR ${number(liveStep.learningRate)}`
      + ` · update ${liveStep.globalStep}`,
      `Members ${populationSize}`
      + ` · losses ${(liveStep.losses ?? []).map(number).join(", ")}`
      + ` · ${(liveStep.jobs ?? []).join(", ")}`,
    );
    if (samePopulation(liveStep, liveEpoch)) {
      const active = liveEpoch.jobs.filter((job) => job.active).length;
      const best = Math.min(...liveEpoch.jobs.map((job) =>
        Number(job.bestValidation)));
      const maximumStale = Math.max(...liveEpoch.jobs.map((job) =>
        Number(job.staleEpochs ?? 0)));
      lines.push(
        `Last completed epoch ${liveEpoch.epoch + 1}`
        + ` · best validation KL ${number(best)}`
        + ` · active ${active}/${liveEpoch.jobs.length}`
        + ` · max stale ${maximumStale}`,
      );
      const deploymentBest = bestDeploymentMetrics(liveEpoch.jobs);
      if (deploymentBest) {
        lines.push(
          `Last deployment KL ${number(deploymentBest.mean)}`
          + ` ± ${number(deploymentBest.standardDeviation)}`,
        );
      }
    }
  } else if (liveEpoch && alive(status?.pid)) {
    const active = liveEpoch.jobs.filter((job) => job.active).length;
    const best = Math.min(...liveEpoch.jobs.map((job) =>
      Number(job.bestValidation)));
    lines.push(
      `Population group ${liveEpoch.populationGroup}/${liveEpoch.populationGroups}`
      + ` · epoch ${liveEpoch.epoch + 1}/${liveEpoch.epochs} complete`
      + ` · ${duration(liveEpoch.seconds)}`
      + ` · best validation KL ${number(best)}`
      + ` · active ${active}/${liveEpoch.jobs.length}`,
    );
    const deploymentBest = bestDeploymentMetrics(liveEpoch.jobs);
    if (deploymentBest) {
      lines.push(
        `Last deployment KL ${number(deploymentBest.mean)}`
        + ` ± ${number(deploymentBest.standardDeviation)}`,
      );
    }
  } else if (groupStart && alive(status?.pid)) {
    lines.push(
      `Population group ${groupStart.populationGroup}/${groupStart.populationGroups}`
      + ` · resuming at epoch ${groupStart.startEpoch + 1}/${groupStart.epochs}`
      + ` · ${groupStart.jobs.length} members`,
    );
  } else if (trainingStart && alive(status?.pid)) {
    lines.push(
      `Population jobs ${trainingStart.jobs - trainingStart.pendingJobs}`
      + `/${trainingStart.jobs} already complete`
      + ` · ${trainingStart.pendingJobs} pending`,
    );
  }
  const fullFidelity = configuration.fidelities?.find(
    (fidelity) => fidelity.key === "full",
  );
  if (status?.fidelity === "full" && fullFidelity?.trainUntilPlateau) {
    lines.push(
      `Plateau rule: patience ${fullFidelity.patience}`
      + ` · minimum KL improvement ${number(fullFidelity.minimumImprovement)}`
      + ` · ceiling ${fullFidelity.maximumEpochs} epochs`,
    );
  }
  if (status?.message) lines.push(status.message);
  if (status?.retention) {
    const retained = status.retention;
    lines.push(
      `Checkpoint retention: ${retained.retainedModels ?? "—"} kept`
      + ` · ${retained.removedModels ?? 0} removed`
      + ` · ${(Number(retained.removedBytes ?? 0) / 2 ** 30).toFixed(2)} GiB reclaimed`,
    );
  }
  if (status?.error) lines.push(`ERROR: ${status.error}`);
  lines.push(
    `Summary: ${summaryFile}`,
    `Log: ${logFile}`,
  );
  process.stdout.write(
    `${watch ? "\x1b[2J\x1b[H" : ""}${lines.join("\n")}\n`,
  );
}

function bestDeploymentMetrics(jobs) {
  const available = jobs
    .map((job) => job.validation)
    .filter((metrics) => Number.isFinite(metrics?.deploymentKlDivergence))
    .sort((left, right) =>
      left.deploymentKlDivergence - right.deploymentKlDivergence);
  if (available.length === 0) return undefined;
  return {
    mean: available[0].deploymentKlDivergence,
    standardDeviation: available[0].deploymentKlDivergenceStdDev,
  };
}

render();
if (watch) setInterval(render, 2_000);
