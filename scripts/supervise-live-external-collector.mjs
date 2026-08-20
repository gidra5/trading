import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runtimeRoot = path.join(repoRoot, "data/runtime-cache/external-live-collector");
const collectorStatusFile = path.join(runtimeRoot, "collector-status.json");
const supervisorStatusFile = path.join(runtimeRoot, "supervisor-status.json");
const stopRequestFile = path.join(runtimeRoot, "stop-request");
const collectorScript = path.join(repoRoot, "scripts/collect-live-external-features.ts");
const watcherScript = path.join(repoRoot, "scripts/watch-live-external-early-screen.ts");
const collectorOutput = path.join(repoRoot, "data/market/mutable/external-live");
const startedAt = Date.now();
let stopping = false;
let child;
let watcher;
let restarts = 0;
let lastExit = null;

await fsp.mkdir(runtimeRoot, { recursive: true });
await fsp.rm(stopRequestFile, { force: true });
await writeStatus();
startWatcher();

process.on("SIGINT", stop);
process.on("SIGTERM", stop);

while (!stopping) {
  child = spawn(process.execPath, [
    "--conditions=development",
    "--import", "tsx",
    collectorScript,
    "--output-dir", collectorOutput,
    "--status-file", collectorStatusFile,
  ], {
    cwd: repoRoot,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
  });
  await writeStatus();
  const exitPromise = new Promise((resolve) => child.once("exit", (code, signal) => resolve({ code, signal })));
  let exit;
  while (!stopping) {
    exit = await Promise.race([exitPromise, delay(5_000).then(() => undefined)]);
    if (exit) break;
    if (fs.existsSync(stopRequestFile)) {
      stopping = true;
      await fsp.rm(stopRequestFile, { force: true });
      child.kill("SIGTERM");
      break;
    }
    const health = readJson(collectorStatusFile);
    const updatedAt = Date.parse(String(health?.updatedAt ?? ""));
    if (!Number.isFinite(updatedAt) || Date.now() - updatedAt > 25_000) {
      lastExit = { at: new Date().toISOString(), reason: "collector heartbeat stale" };
      child.kill("SIGTERM");
      exit = await exitPromise;
      break;
    }
    await writeStatus();
  }
  if (stopping) break;
  lastExit = { at: new Date().toISOString(), ...(exit ?? {}), reason: lastExit?.reason ?? "collector exited" };
  restarts += 1;
  await writeStatus();
  await delay(Math.min(30_000, 1_000 * 2 ** Math.min(5, restarts - 1)));
}

if (child && child.exitCode === null) {
  child.kill("SIGTERM");
  await Promise.race([new Promise((resolve) => child.once("exit", resolve)), delay(10_000)]);
  if (child.exitCode === null) child.kill("SIGKILL");
}
if (watcher && watcher.exitCode === null) watcher.kill("SIGTERM");
await writeStatus();

function stop() {
  stopping = true;
  if (child && child.exitCode === null) child.kill("SIGTERM");
  if (watcher && watcher.exitCode === null) watcher.kill("SIGTERM");
}

function startWatcher() {
  if (stopping || (watcher && watcher.exitCode === null)) return;
  watcher = spawn(process.execPath, [
    "--conditions=development",
    "--import", "tsx",
    watcherScript,
    "--checkpoints", "1,24,72,168",
    "--poll-seconds", "900",
  ], {
    cwd: repoRoot,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
  });
  watcher.once("exit", (code) => {
    if (!stopping && code !== 0) setTimeout(startWatcher, 5_000).unref();
  });
}

async function writeStatus() {
  const value = {
    version: 1,
    pid: process.pid,
    startedAt: new Date(startedAt).toISOString(),
    updatedAt: new Date().toISOString(),
    state: stopping ? "stopping" : "running",
    childPid: child?.pid ?? null,
    watcherPid: watcher?.exitCode === null ? watcher.pid : null,
    restarts,
    lastExit,
  };
  const temporary = `${supervisorStatusFile}.${process.pid}.tmp`;
  await fsp.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await fsp.rename(temporary, supervisorStatusFile);
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return undefined; }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
