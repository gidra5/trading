/** Reconstruct saved native-screen populations and diagnose frozen laws. No fitting. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventMarginalCrps } from "../packages/bot-algo/src/event-crps.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventCashHorizon } from "../packages/bot-algo/src/event-cash-horizon.js";
import { loadNativeEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), output = path.resolve(root, "data/benchmarks", arg("output"));
const estimationMode = arg("estimation-mode");
assert.ok(arg("sources") && arg("output") && !fs.existsSync(output));
assert.ok(["chain", "stride"].includes(estimationMode), "Specify the saved later-fit estimation mode");
fs.mkdirSync(output, { recursive: true });
const save = (file: string, data: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(data, null, 2));
const hash = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
save("method.json", {
  sources: arg("sources").split(","), estimationMode,
  method: "Frozen saved models, unchanged costs. Reconstruct the earlier partition, final-fit-day estimation, following diagnostic calibration and test-chain populations. Verify source-reference hashes and sample counts. Earliest-finish interval scheduling counts the maximum non-overlapping label intervals with a shared boundary allowed; this is not statistical independence or disjoint feature support. CRPS scores physical return and duration marginals, not their dependence. Calibration replay uses exact marked H1 with next-open fills and actual terminal fees. No fitting or selection in this audit.",
});
save("sources.json", Object.fromEntries(["scripts/audit-native-event-forecasts.ts", "scripts/research-event-policy.ts",
  "scripts/event-fit-periods.ts",
  "packages/bot-algo/src/event-crps.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-cash-horizon.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const results = [];
for (const name of arg("sources").split(",")) {
  const source = path.resolve(root, "data/benchmarks", name);
  const read = (file: string) => JSON.parse(fs.readFileSync(path.join(source, file), "utf8"));
  const config = read("config.json"), original = read("summary.json"), modelBytes = fs.readFileSync(path.join(source, "model.json"));
  assert.equal(config.contract, "native-second-event-screen-v1");
  assert.ok(config.estimation.startsWith("Partition on earlier fit days"));
  const policy = restoreEventPolicy(JSON.parse(modelBytes.toString()));
  assert.deepEqual(policy.model.clock, config.clock); assert.deepEqual(policy.costs, config.costs);
  for (const ref of config.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
  const candles = loadNativeEventCandles([
    { start: config.fitStart - (config.warmupCandles + 1) * 1000, end: config.calibrationEnd },
    { start: config.start - (config.warmupCandles + 1) * 1000, end: config.end },
  ]);
  const sample = (start: number, end: number, excluded: typeof config.excluded, mode: "chain" | "stride") =>
    makeSamples(candles, policy.model.clock, start, end, excluded, mode === "chain" ? 1 : config.stride, mode, policy.model.featureNames);
  const populations = {
    training: sample(config.fitStart, config.fitEnd - 86400000, config.excluded, "stride"),
    estimation: sample(config.fitEnd - 86400000, config.fitEnd, config.excluded, estimationMode as "chain" | "stride"),
    calibration: sample(config.calibrationStart, config.calibrationEnd, config.excluded, "stride"),
    test: sample(config.start, config.end, [], "chain"),
  };
  const describe = (rows: MoveSample[]) => {
    const nonOverlapping = (group: MoveSample[]) => {
      let end = -Infinity, count = 0;
      for (const row of group.slice().sort((a, b) => a.end - b.end || a.start - b.start))
        if (row.start >= end) { count++; end = row.end; }
      return count;
    };
    return { count: rows.length, ...eventMarginalCrps(policy.model, rows),
      physicalTargetsHash: hash(JSON.stringify(rows.map(s => [candles[s.start].openTime, candles[s.end].openTime, s.return, s.duration, s.low, s.high]))),
      leaves: policy.model.kernels.map((kernel, leaf) => {
        const group = rows.filter(s => eventLeaf(policy.model, s.features) === leaf);
        return { leaf, count: group.length, maxNonOverlappingLabels: nonOverlapping(group),
          forecastMeanBps: kernel.reduce((s, a) => s + a.return * a.probability * 10000, 0),
          observedMeanBps: group.length ? group.reduce((s, a) => s + a.return * 10000, 0) / group.length : null };
      }),
    };
  };
  for (const [key, rows] of Object.entries(populations)) assert.equal(rows.length, original[key]);
  assert.ok(populations.training.every(s => s.end < populations.estimation[0].start));
  const { trace, positions, ...calibrationReplay } = replayEventPolicy(candles, policy, config.calibrationStart, config.calibrationEnd, 1,
    { trace: true, oneStepTerminal: "marked" });
  save(`${name}-calibration-trades.json`, trace); save(`${name}-calibration-positions.json`, positions);
  const bounds = eventCashHorizon(policy.model, policy.costs, 100);
  results.push({ name, modelHash: hash(modelBytes), configHash: hash(fs.readFileSync(path.join(source, "config.json"))),
    nodesHash: hash(JSON.stringify(policy.model.nodes)), kernelsHash: hash(JSON.stringify(policy.model.kernels)),
    populations: Object.fromEntries(Object.entries(populations).map(([key, rows]) => [key, describe(rows)])),
    calibrationReplay, testReplay: original.metrics, cashBoundDepth: bounds.verifiedDepth });
}
save("summary.json", { results });
console.log(JSON.stringify(results.map(r => ({ name: r.name, calibrationReturnPct: r.calibrationReplay.returnPct,
  testReturnPct: r.testReplay.returnPct, calibration: r.populations.calibration, cashBoundDepth: r.cashBoundDepth }))));
