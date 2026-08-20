import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runtimeRoot = path.join(repoRoot, "data/runtime-cache/external-live-collector");
const supervisorStatusFile = path.join(runtimeRoot, "supervisor-status.json");
const collectorStatusFile = path.join(runtimeRoot, "collector-status.json");
const stopRequestFile = path.join(runtimeRoot, "stop-request");
const command = process.argv[2] ?? "status";

if (command === "start") await start();
else if (command === "stop") await stop();
else if (command === "status") status();
else throw new Error("Usage: node scripts/manage-live-external-collector.mjs start|status|stop");

async function start() {
  const previous = readJson(supervisorStatusFile);
  if (isAlive(previous?.pid)) {
    console.log(`Live external collector supervisor is already running (PID ${previous.pid}).`);
    status();
    return;
  }
  await fsp.mkdir(runtimeRoot, { recursive: true });
  await fsp.rm(stopRequestFile, { force: true });
  const output = fs.openSync(path.join(runtimeRoot, "collector.log"), "a");
  const child = spawn(process.execPath, [path.join(repoRoot, "scripts/supervise-live-external-collector.mjs")], {
    cwd: repoRoot,
    detached: true,
    windowsHide: true,
    stdio: ["ignore", output, output],
    env: process.env,
  });
  child.unref();
  fs.closeSync(output);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await delay(250);
    const current = readJson(supervisorStatusFile);
    if (current?.pid === child.pid && isAlive(current.pid)) {
      for (let collectorAttempt = 0; collectorAttempt < 40; collectorAttempt += 1) {
        const collector = readJson(collectorStatusFile);
        if (collector?.pid === current.childPid && isAlive(collector.pid)) {
          console.log(`Started live external collector supervisor (PID ${child.pid}).`);
          status();
          return;
        }
        await delay(250);
      }
      throw new Error(`Supervisor PID ${child.pid} started, but collector PID ${current.childPid} did not publish status.`);
    }
  }
  throw new Error(`Supervisor PID ${child.pid} did not publish healthy status; inspect data/runtime-cache/external-live-collector/collector.log`);
}

async function stop() {
  const current = readJson(supervisorStatusFile);
  if (!isAlive(current?.pid)) {
    console.log("Live external collector supervisor is not running.");
    return;
  }
  await fsp.writeFile(stopRequestFile, `${new Date().toISOString()}\n`, "utf8");
  for (let attempt = 0; attempt < 80 && isAlive(current.pid); attempt += 1) await delay(250);
  if (isAlive(current.pid)) throw new Error(`Supervisor PID ${current.pid} did not stop within 20 seconds.`);
  console.log(`Stopped live external collector supervisor (PID ${current.pid}).`);
}

function status() {
  const supervisor = readJson(supervisorStatusFile);
  const collector = readJson(collectorStatusFile);
  console.log(JSON.stringify({
    running: isAlive(supervisor?.pid),
    supervisor,
    collector,
  }, null, 2));
}

function isAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch { return false; }
}

function readJson(file) {
  try { return JSON.parse(fs.readFileSync(file, "utf8")); } catch { return undefined; }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
