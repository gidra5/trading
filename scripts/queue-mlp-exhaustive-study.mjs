import fs from "node:fs";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const WAIT_MS = 30_000;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const planFile = path.resolve(repoRoot, "ml/training-plan.json");
const plan = JSON.parse(fs.readFileSync(planFile, "utf8"));
const source = plan.lossWeightStudy;
const exhaustive = plan.exhaustiveLossWeightStudy;
if (!source?.runDir || !source?.outputDir || !exhaustive?.runDir) {
  throw new Error("Both fractional and exhaustive loss-weight studies must be configured.");
}
const sourceStatusFile = path.resolve(repoRoot, source.runDir, "status.json");
const exhaustiveStatusFile = path.resolve(repoRoot, exhaustive.runDir, "status.json");
const queueFile = path.resolve(repoRoot, exhaustive.runDir, "queue.json");

for (;;) {
  const exhaustiveStatus = readJson(exhaustiveStatusFile);
  if (exhaustiveStatus?.pid && processIsAlive(exhaustiveStatus.pid)
    && !["complete", "failed", "paused", "review"].includes(exhaustiveStatus.stage)) {
    writeQueue({
      stage: "already-running",
      exhaustivePid: exhaustiveStatus.pid,
      exhaustiveStage: exhaustiveStatus.stage,
    });
    break;
  }
  if (["complete", "review"].includes(exhaustiveStatus?.stage)) {
    writeQueue({
      stage: exhaustiveStatus.stage,
      exhaustivePid: exhaustiveStatus.pid,
      message: "The exhaustive study already reached its configured terminal gate.",
    });
    break;
  }
  const sourceStatus = readJson(sourceStatusFile);
  if (sourceStatus?.stage !== "complete") {
    writeQueue({
      stage: "waiting-for-fractional-study",
      sourcePid: sourceStatus?.pid,
      sourceStage: sourceStatus?.stage ?? "not-started",
      sourceCompletedRuns: sourceStatus?.completedRuns ?? 0,
      sourcePlannedRuns: sourceStatus?.variants,
    });
    await wait(WAIT_MS);
    continue;
  }

  writeQueue({ stage: "pruning-fractional-artifacts" });
  const retention = spawnSync(process.execPath, [
    path.join(repoRoot, "scripts/mlp-study-retention.mjs"),
    path.resolve(repoRoot, source.outputDir),
    "--best-per-delay",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (retention.status !== 0) {
    throw new Error(
      `Could not retain fractional winners: ${retention.stderr || retention.stdout}`,
    );
  }

  const child = spawn(process.execPath, [
    path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
    path.join(repoRoot, "scripts/run-mlp-joint-study.mjs"),
    "--study",
    "exhaustiveLossWeightStudy",
  ], {
    cwd: repoRoot,
    detached: true,
    stdio: "ignore",
  });
  child.unref();
  writeQueue({
    stage: "launched",
    launcherPid: child.pid,
    retention: parseLastJson(retention.stdout),
  });
  break;
}

function writeQueue(value) {
  const status = {
    ...value,
    pid: process.pid,
    updatedAt: new Date().toISOString(),
    sourceStatusFile,
    exhaustiveStatusFile,
  };
  fs.mkdirSync(path.dirname(queueFile), { recursive: true });
  const temporary = `${queueFile}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(status, null, 2)}\n`);
  fs.renameSync(temporary, queueFile);
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT" || error instanceof SyntaxError) return undefined;
    throw error;
  }
}

function parseLastJson(value) {
  const line = value.trim().split("\n").at(-1);
  try {
    return JSON.parse(line);
  } catch {
    return { output: value.trim() };
  }
}

function processIsAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
