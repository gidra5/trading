import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { run } from "./analyze-live-external-early-screen.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const artifactFile = path.join(repoRoot, "data/benchmarks/live-external-early-screen.json");
const stateFile = path.join(repoRoot, "data/benchmarks/live-external-early-screen-watch.json");

interface State {
  pid: number;
  startedAt: string;
  collectorStartedAt: string;
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
  const checkpoints = (value("--checkpoints") ?? "1,24")
    .split(",")
    .map(Number)
    .filter((item) => Number.isFinite(item) && item > 0)
    .sort((left, right) => left - right);
  if (checkpoints.length === 0) throw new Error("At least one positive checkpoint hour is required.");
  if (!fs.existsSync(artifactFile)) run();
  const artifact = JSON.parse(fs.readFileSync(artifactFile, "utf8")) as { collectorStartedAt: string };
  const state: State = {
    pid: process.pid,
    startedAt: new Date().toISOString(),
    collectorStartedAt: artifact.collectorStartedAt,
    checkpointsHours: checkpoints,
    completedHours: [],
    status: "running",
  };
  writeState(state);
  try {
    for (const checkpoint of checkpoints) {
      const dueAt = Date.parse(state.collectorStartedAt) + checkpoint * 3_600_000;
      while (Date.now() < dueAt) await delay(Math.min(60_000, dueAt - Date.now()));
      run(["--checkpoint", `${checkpoint}h`]);
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
