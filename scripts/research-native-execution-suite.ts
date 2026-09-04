/** Bounded computation gate followed by complete fixed-law execution-H1 replays. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const config = read(path.join(source, "config.json")), summary = read(path.join(source, "summary.json"));
const requested = arg("windows").split(",").filter(Boolean), excluded = new Set(arg("exclude").split(",").filter(Boolean));
assert.ok(requested.every(id => config.catalog.some((w: any) => w.id === id)));
assert.ok([...excluded].every(id => config.catalog.some((w: any) => w.id === id)));
const windows = config.catalog.filter((w: any) => !excluded.has(w.id) && (!requested.length || requested.includes(w.id)));
assert.ok(windows.length && windows.every((w: any) => !w.id.startsWith("fit-") && w.id !== "latest"));
fs.mkdirSync(output, { recursive: true });
const save = (name: string, data: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(data, null, 2));
save("config.json", { contract: "native-event-execution-suite-v1", source, windows,
  sourceSummaryHash: createHash("sha256").update(fs.readFileSync(path.join(source, "summary.json"))).digest("hex"),
  method: "Preserve saved fitted laws. Compile exact empirical execution paths, profile every predeclared calibration leaf/held account, and run the complete inspector interval only when all H1 searches are finite, complete and under two seconds. Returns never select the window or gate. Each completed replay receives an independent value/transition audit. This is execution-aware H1 coverage, not deeper Bellman convergence or forecast promotion." });
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename, "utf8"));
const results: any[] = [], started = performance.now();
const run = (id: string, phase: string, script: string, args: string[]) => {
  const folder = path.join(output, id); fs.mkdirSync(folder, { recursive: true });
  save("progress.json", { id, phase, processed: results.length, requested: windows.length });
  const log = fs.openSync(path.join(folder, `${phase}.log`), "w");
  const child = spawnSync(process.execPath, ["--conditions=development", "--import", "tsx", script, ...args],
    { cwd: root, windowsHide: true, stdio: ["ignore", log, log] });
  fs.closeSync(log);
  if (child.status !== 0) throw new Error(`${phase} exited ${child.status}: ${child.error?.message ?? path.join(folder, `${phase}.log`)}`);
};
for (const window of windows) {
  const id = window.id, before = performance.now();
  const law = path.join(output, id, "law"), profile = path.join(output, id, "profile"), replay = path.join(output, id, "full"), audit = path.join(output, id, "audit");
  try {
    const screen = summary.results.find((r: any) => r.window.id === id); assert.equal(screen.status, "screened");
    run(id, "compile", "scripts/compile-native-event-execution-law.ts", ["--source", path.join(source, id, "law"), "--output", law]);
    run(id, "profile", "scripts/probe-native-event-execution.ts", ["--source", law, "--output", profile, "--count", "all"]);
    const profiles = read(path.join(profile, "summary.json")).results;
    if (!profiles.every((r: any) => r.result.complete && r.result.feasible && r.solveSeconds < 2)) {
      results.push({ window, status: "profile-unresolved", law, profile });
    } else {
      const availability = id === "sideways-churn-2023-03" ? ["--availability", path.join(root,
        "data/benchmarks/event-native-march-availability-checks-v477/availability.json")] : [];
      run(id, "replay", "scripts/replay-native-event-execution.ts", ["--source", law, "--output", replay, "--phase", "test", ...availability]);
      run(id, "audit", "scripts/audit-native-execution-policy.ts", ["--replays", replay, "--output", audit]);
      const audited = read(path.join(audit, "summary.json")).results[0];
      assert.equal(audited.modelHash, screen.modelHash);
      results.push({ window, status: "complete-h1", law, profile, replay, audit, decisions: audited.decisions,
        controlReturnPct: audited.control.returnPct, executionReturnPct: audited.execution.returnPct,
        executionSeconds: audited.executionSeconds, elapsedSeconds: (performance.now() - before) / 1000 });
    }
  } catch (error) { results.push({ window, status: "failed", law, profile, replay, audit, error: String(error) }); }
  save("summary.json", { requested: windows.length, processed: results.length, results, elapsedSeconds: (performance.now() - started) / 1000 });
  console.log(JSON.stringify(results.at(-1)));
}
save("progress.json", { phase: "finished", processed: results.length, requested: windows.length });
if (results.some(r => r.status !== "complete-h1")) process.exitCode = 1;
