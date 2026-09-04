/** Sequential native coverage; child artifacts retain full reproducibility. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import { DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";
import { NATIVE_SECOND_CONTEXT_FEATURES } from "../packages/bot-algo/src/event-second-features.js";

const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const outputName = arg("output"), output = directory(outputName), mode = arg("mode", "screen"), sourceName = arg("source");
assert.ok(outputName && !fs.existsSync(output), "Specify a new --output directory");
assert.ok(["screen", "replay"].includes(mode) && (mode === "replay" ? sourceName : !sourceName));
const budget = Number(arg("budget", "512")); assert.ok(Number.isInteger(budget) && budget >= 2);
const catalog = new KamaInspector(path.join(root, "data")).catalog().windows.filter(w => w.id !== "latest" && !w.id.startsWith("fit-"));
const requested = arg("windows", "all").split(","), windows = catalog.filter(w => requested[0] === "all" || requested.includes(w.id));
assert.ok(windows.length && (requested[0] === "all" || requested.every(id => windows.some(w => w.id === id))));
const hash = (data: Buffer | string) => createHash("sha256").update(data).digest("hex");
const read = (name: string, file: string) => JSON.parse(fs.readFileSync(path.join(directory(name), file), "utf8"));
let sourceConfigHash: string | undefined;
if (mode === "replay") {
  const bytes = fs.readFileSync(path.join(directory(sourceName), "config.json")), config = JSON.parse(bytes.toString());
  assert.equal(config.contract, "native-event-coverage-suite-v1"); assert.equal(config.mode, "screen"); sourceConfigHash = hash(bytes);
}
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("config.json", { contract: "native-event-coverage-suite-v1", mode, sourceName, sourceConfigHash, windows, catalog, budget,
  protocol: { candleIntervalMs: 1000, thresholdBps: 48, maxCandles: 3600, features: NATIVE_SECOND_CONTEXT_FEATURES,
    fitDays: 4, estimation: "later-fit-stride", extraEstimationDays: 0, weighting: "uniqueness", priorMode: "matched-ratio", costs: DEFAULT_EVENT_COSTS },
  maxProbeSeconds: 2,
  method: "Screen all requested windows with the same date-only admissible fit and separate calibration, native 48bp/3600s events and frozen uniqueness-weighted joint law. Replay consumes saved laws and profiles every first calibration leaf plus first held account. Scale only when all probes meet 0.001bp and each takes <=2s. Returns and accuracy never select coverage. Preserve budget stops and data failures and continue other windows. Child processes run sequentially with saved logs and complete source/model artifacts. Full replay retains next-open execution; certificates concern finite decision-price marked-terminal H2, not stationary or execution-optimal control, independent holdouts or profitability." });
save("sources.json", Object.fromEntries(["scripts/audit-native-event-suite.ts", "scripts/research-native-second-events.ts",
  "scripts/reestimate-native-event-law.ts", "scripts/research-native-event-planning.ts", "scripts/audit-native-event-planning.ts",
  "scripts/event-fit-periods.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results: any[] = [];
const checkpoint = () => save("summary.json", { mode, requestedWindows: windows.length, catalogWindows: catalog.length,
  processedWindows: results.length, screened: results.filter(r => r.status === "screened").length,
  certifiedFullH2Windows: results.filter(r => r.status === "certified").length, failures: results.filter(r => r.status === "failed").length,
  unresolved: results.filter(r => ["uncertified", "probe-unresolved"].includes(r.status)).length,
  elapsedSeconds: (performance.now() - started) / 1000, results });
const run = (id: string, phase: string, script: string, args: string[]) => {
  const log = path.join(output, id, `${phase}.log`);
  save("progress.json", { window: id, phase, status: "started", command: [process.execPath, script, ...args] });
  const fd = fs.openSync(log, "w"); let result;
  try { result = spawnSync(process.execPath, ["--conditions=development", "--import", "tsx", script, ...args],
    { cwd: root, stdio: ["ignore", fd, fd], windowsHide: true }); } finally { fs.closeSync(fd); }
  if (result.error || result.status !== 0) throw new Error(`${phase} failed (${result.status ?? result.signal}): ${result.error?.message ?? fs.readFileSync(log, "utf8").slice(-2500)}`);
};
for (const window of windows) {
  const begin = performance.now(), prefix = `${outputName}/${window.id}`;
  fs.mkdirSync(directory(prefix), { recursive: true }); let phase = "fit";
  try {
    if (mode === "screen") {
      run(window.id, phase, "scripts/research-native-second-events.ts", ["--window", window.id, "--clock", "barrier",
        "--threshold-bps", "48", "--max-candles", "3600", "--test-hours", "24", "--features", "context",
        "--fit-days", "4", "--estimation", "later-fit-stride", "--output", `${prefix}/fit`]);
      phase = "law";
      run(window.id, phase, "scripts/reestimate-native-event-law.ts", ["--source", `${prefix}/fit`, "--extra-days", "0",
        "--sampling", "stride", "--weighting", "uniqueness", "--prior-mode", "matched-ratio", "--output", `${prefix}/law`]);
      const config = read(`${prefix}/law`, "config.json"), summary = read(`${prefix}/law`, "summary.json");
      assert.deepEqual(config.window, window); assert.deepEqual(config.costs, DEFAULT_EVENT_COSTS);
      assert.deepEqual(config.featureNames, NATIVE_SECOND_CONTEXT_FEATURES);
      results.push({ window, status: "screened", law: `${prefix}/law`, modelHash: hash(fs.readFileSync(path.join(directory(`${prefix}/law`), "model.json"))),
        fitPeriods: config.fitPeriods, training: summary.training, estimation: summary.estimation, estimationMass: summary.estimationMass,
        cashBoundDepth: summary.cashBoundDepth, calibrationReturnPct: summary.calibrationMetrics.returnPct,
        testPrefixReturnPct: summary.metrics.returnPct, elapsedSeconds: (performance.now() - begin) / 1000 });
    } else {
      phase = "verify-source";
      const entry = read(sourceName, "summary.json").results.find((r: any) => r.window.id === window.id);
      assert.ok(entry?.status === "screened", "No completed source screen for this window");
      const source = entry.law, config = read(source, "config.json"), bytes = fs.readFileSync(path.join(directory(source), "model.json"));
      assert.equal(hash(bytes), entry.modelHash); assert.deepEqual(config.window, window);
      assert.deepEqual(config.costs, DEFAULT_EVENT_COSTS); assert.deepEqual(config.featureNames, NATIVE_SECOND_CONTEXT_FEATURES);
      assert.ok(config.calibrationEnd <= window.startTime);
      const calibration = read(source, "calibration-trades.json"), seen = new Set<number>(), probes: any[] = [];
      for (const row of calibration) if (!seen.has(row.leaf)) { seen.add(row.leaf); probes.push(row); }
      const held = calibration.find((row: any) => row.exposureBefore !== 0);
      if (held && !probes.includes(held)) probes.push(held);
      phase = "probe";
      run(window.id, phase, "scripts/research-native-event-planning.ts", ["--source", source, "--phase", "calibration", "--mode", "probe",
        "--count", String(probes.length), "--budget", String(budget), "--global-upper", "true", "--output", `${prefix}/probe`]);
      const measured = read(`${prefix}/probe`, "summary.json").results;
      if (measured.some((r: any) => !r.h2.converged || r.elapsedSeconds > 2)) {
        results.push({ window, status: "probe-unresolved", source, modelHash: entry.modelHash, probe: `${prefix}/probe`,
          probes: measured.map((r: any) => ({ leaf: r.leaf, gapBps: r.h2.gap * 10000, converged: r.h2.converged, elapsedSeconds: r.elapsedSeconds })),
          elapsedSeconds: (performance.now() - begin) / 1000 });
      } else {
        phase = "full";
        run(window.id, phase, "scripts/research-native-event-planning.ts", ["--source", source, "--phase", "test", "--mode", "replay",
          "--budget", String(budget), "--global-upper", "true", "--full-window", "--output", `${prefix}/full`]);
        const summary = read(`${prefix}/full`, "summary.json"), r = summary.results[0], certified = r.decisions === r.converged;
        if (certified) {
          phase = "audit";
          run(window.id, phase, "scripts/audit-native-event-planning.ts", ["--replays", `${prefix}/full`, "--output", `${prefix}/audit`]);
        }
        results.push({ window, status: certified ? "certified" : "uncertified", source, modelHash: entry.modelHash,
          replay: `${prefix}/full`, audit: certified ? `${prefix}/audit` : undefined, decisions: r.decisions, converged: r.converged,
          maxGapBps: r.maxGapBps, h1ReturnPct: summary.h1.returnPct, h2ReturnPct: r.metrics.returnPct,
          h2DrawdownPct: r.metrics.maxDrawdownPct, h2Orders: r.metrics.trades, h2Cancellations: r.metrics.canceledOrders,
          elapsedSeconds: (performance.now() - begin) / 1000 });
      }
    }
  } catch (error) {
    const failure = { window, status: "failed", phase, error: error instanceof Error ? error.message : String(error),
      elapsedSeconds: (performance.now() - begin) / 1000 };
    results.push(failure); save(`${window.id}/failure.json`, failure);
  }
  checkpoint(); console.log(JSON.stringify(results.at(-1)));
}
save("progress.json", { status: "finished", processed: results.length });
