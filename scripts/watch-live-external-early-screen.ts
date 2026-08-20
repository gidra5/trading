import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { run } from "./analyze-live-external-early-screen.ts";
import { run as runFast } from "./analyze-live-fast-features.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifactFile = path.join(repoRoot, "data/benchmarks/live-external-early-screen.json");
const stateFile = path.join(repoRoot, "data/benchmarks/live-external-early-screen-watch.json");

interface State {
  pid: number;
  startedAt: string;
  collectorStartedAt: string;
  observedCoverageHours: number;
  checkpointsHours: number[];
  completedHours: number[];
  status: "running" | "complete" | "failed";
  error?: string;
}

export async function watch(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const checkpoints = (value("--checkpoints") ?? "1,24,72,168")
    .split(",")
    .map(Number)
    .filter((item) => Number.isFinite(item) && item > 0)
    .sort((left, right) => left - right);
  if (checkpoints.length === 0) throw new Error("At least one positive checkpoint hour is required.");
  const pollSeconds = Math.max(10, Number(value("--poll-seconds") ?? 900));
  if (!fs.existsSync(artifactFile)) run();
  const artifact = JSON.parse(fs.readFileSync(artifactFile, "utf8")) as { collectorStartedAt: string };
  const state: State = {
    pid: process.pid,
    startedAt: new Date().toISOString(),
    collectorStartedAt: artifact.collectorStartedAt,
    observedCoverageHours: 0,
    checkpointsHours: checkpoints,
    completedHours: [],
    status: "running",
  };
  writeState(state);
  try {
    for (const checkpoint of checkpoints) {
      let current = run() as { durationHours: number; collectorStartedAt: string };
      state.collectorStartedAt = current.collectorStartedAt;
      state.observedCoverageHours = current.durationHours;
      writeState(state);
      while (current.durationHours < checkpoint) {
        await delay(pollSeconds * 1_000);
        current = run() as { durationHours: number; collectorStartedAt: string };
        state.collectorStartedAt = current.collectorStartedAt;
        state.observedCoverageHours = current.durationHours;
        writeState(state);
      }
      run(["--checkpoint", `${checkpoint}h`]);
      if (checkpoint >= 24) runFast([
        "--output", `data/benchmarks/live-fast-feature-early-screen-${checkpoint}h.json`,
        "--report", `docs/experiments/live-fast-feature-early-screen-2026-08-19-${checkpoint}h.md`,
      ]);
      state.completedHours.push(checkpoint);
      writeState(state);
    }
    state.status = "complete";
    writeState(state);
  } catch (error) {
    state.status = "failed";
    state.error = error instanceof Error ? error.stack ?? error.message : String(error);
    writeState(state);
    throw error;
  }
}

function writeState(state: State) {
  fs.mkdirSync(path.dirname(stateFile), { recursive: true });
  fs.writeFileSync(stateFile, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function delay(milliseconds: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, milliseconds));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  void watch().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
