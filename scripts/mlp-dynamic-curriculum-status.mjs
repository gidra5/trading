import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const plan = JSON.parse(fs.readFileSync(path.join(repoRoot, "ml/training-plan.json"), "utf8"));
const configuration = plan.dynamicCurriculumStudy;
const runDir = path.resolve(repoRoot, configuration.runDir);
const outputDir = path.resolve(repoRoot, configuration.outputDir);
const statusFile = path.join(runDir, "status.json");
const summaryFile = path.join(outputDir, "summary.json");
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

function render() {
  const status = readJson(statusFile);
  const summary = readJson(summaryFile);
  const stages = configuration.delayScheduleMinutes.length;
  const completed = summary?.completedStages ?? 0;
  const best = summary?.bestObserved?.validation;
  const lines = [
    `${plan.id} · dynamic delay/weight curriculum`,
    status
      ? `${status.stage ?? "unknown"} · updated ${status.updatedAt ?? "—"}`
      : "not started",
    `PID ${status?.pid ?? "—"} · ${alive(status?.pid) ? "running" : "not running"}`,
    `Stage ${status?.curriculumStage ?? completed}/${stages}`
      + (Number.isFinite(status?.delayMinutes) ? ` · delay ${status.delayMinutes}m` : ""),
  ];
  if (Number.isFinite(status?.branches)) {
    lines.push(`${status.parents} parents · ${status.branches} branches`);
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
  if (status?.message) lines.push(status.message);
  if (status?.error) lines.push(`ERROR: ${status.error}`);
  lines.push(`Summary: ${summaryFile}`, `Log: ${path.join(runDir, "study.log")}`);
  process.stdout.write(`${watch ? "\x1b[2J\x1b[H" : ""}${lines.join("\n")}\n`);
}

render();
if (watch) setInterval(render, 2_000);
