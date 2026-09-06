/** Screen finite-horizon convergence on the existing interpolated Bellman grid. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { buildEventPolicy, decideEvent, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";

const argument = (name: string, fallback = "") => {
  const at = process.argv.indexOf(`--${name}`);
  return at < 0 ? fallback : process.argv[at + 1];
};
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.join(root, "data/benchmarks", name);
const source = directory(argument("source", "event-policy-all-mean120-1x-cap-v10"));
const h3Source = directory(argument("h3-source", "event-policy-three-event-full-suite-v390"));
const output = directory(argument("output"));
const maxDepth = Number(argument("depth", "64"));
const actionSteps = argument("action-steps", "5,10,20").split(",").map(Number);
const requestedWindows = argument("windows", "all").split(",");
if (!argument("output")) throw new Error("Specify a new --output directory");
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
if (!Number.isInteger(maxDepth) || maxDepth < 4 || maxDepth > 512
  || !actionSteps.length || actionSteps.some(v => !Number.isInteger(v) || v < 1 || v > 100)
  || new Set(actionSteps).size !== actionSteps.length) throw new Error("Invalid horizon convergence grid");
const read = (d: string, file: string) => JSON.parse(fs.readFileSync(path.join(d, file), "utf8"));
const sourceConfig = read(source, "config.json"), sourceSummary = read(source, "summary.json");
const h3Config = read(h3Source, "config.json"), h3Summary = read(h3Source, "summary.json");
assert.equal(sourceConfig.contract, "causal-event-tree-bellman-v1");
assert.equal(h3Config.contract, "event-three-step-suite-merge-v1");
assert.equal(path.resolve(h3Config.reference), directory("event-policy-two-event-full-suite-merged-v309"));
assert.equal(h3Summary.completeCoverage, true); assert.equal(h3Summary.catalogWindows, h3Summary.windows);
const sourceRows = Array.isArray(sourceSummary) ? sourceSummary : sourceSummary.results;
assert.deepEqual(sourceRows.map((r: any) => r.window.id).sort(), h3Summary.results.map((r: any) => r.window.id).sort());
assert.ok(requestedWindows[0] === "all" || requestedWindows.every(id => h3Summary.results.some((r: any) => r.window.id === id)));
const selected = h3Summary.results.filter((r: any) => requestedWindows[0] === "all" || requestedWindows.includes(r.window.id));

fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-grid-horizon-convergence-v1", source, h3Source, maxDepth, actionSteps,
  windows: selected.map((r: any) => r.window.id),
  terminal: "marked",
  method: "For every frozen forecast in the complete non-fit inspector suite, rebuild the existing finite equity/price/exposure interpolation grid with a marked terminal boundary and iterate the Bellman operator to the declared depth. Compare H3 with the final finite-horizon action at all grid states and at every account visited by the certified exact H3 replay. Repeat at several action-grid resolutions. This is a cheap convergence screen for the fitted finite-state approximation; it is not an exchange-lattice H4 certificate or proof of the continuous stationary optimum." });
save("sources.json", Object.fromEntries([
  "scripts/audit-event-grid-horizon-convergence.ts",
  "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));

const results: any[] = [], started = performance.now();
for (const h3 of selected) {
  const filename = `${h3.window.id}-model.json`, bytes = fs.readFileSync(path.join(source, filename));
  assert.equal(createHash("sha256").update(bytes).digest("hex"), h3.modelHash);
  const saved = JSON.parse(bytes.toString()), base = restoreEventPolicy(saved.policy);
  const trace = read(h3.h3Source, `${h3.window.id}-trades.json`);
  assert.equal(trace.length, h3.decisions);
  for (const steps of actionSteps) {
    const begin = performance.now(), policy = buildEventPolicy(base.model, base.costs, { depths: maxDepth,
      referenceEquity: base.equities[2], referencePrice: base.prices[1], actionSteps: steps, terminal: "marked" });
    const convergence = policy.tables.map(t => ({ depth: t.depth, ...t.convergence }));
    const changed = convergence.filter(r => r.depth > 1 && r.changedActionFraction! > 0);
    const grid: any[] = [];
    for (let leaf = 0; leaf < base.model.kernels.length; leaf++) for (const equity of policy.equities)
      for (const price of policy.prices) for (const exposure of policy.exposures) {
        const account = { equity, price, exposure }, h3Decision = decideEvent(policy, leaf, account, 3);
        const finalDecision = decideEvent(policy, leaf, account, maxDepth);
        grid.push({ leaf, equity, price, exposure, h3Exposure: h3Decision.exposure,
          finalExposure: finalDecision.exposure, difference: finalDecision.exposure - h3Decision.exposure });
      }
    const visited = trace.map((row: any, i: number) => {
      const account = { equity: row.equityBefore, price: row.order.price, exposure: row.exposureBefore };
      const h3Grid = decideEvent(policy, row.leaf, account, 3), final = decideEvent(policy, row.leaf, account, maxDepth);
      return { decision: i + 1, time: row.time, leaf: row.leaf, account,
        exactH3Exposure: row.order.exposure, gridH3Exposure: h3Grid.exposure, finalExposure: final.exposure,
        h3ToFinalDifference: final.exposure - h3Grid.exposure,
        exactH3ToFinalDifference: final.exposure - row.order.exposure };
    });
    const different = grid.filter(r => Math.abs(r.difference) > 1e-8), visitedDifferent = visited.filter(r => Math.abs(r.h3ToFinalDifference) > 1e-8);
    const result = { window: h3.window, modelHash: h3.modelHash, actionSteps: steps, maxDepth,
      leaves: base.model.kernels.length, gridStates: grid.length,
      lastActionChangeDepth: changed.at(-1)?.depth ?? 1,
      stableActionFromDepth: convergence.at(-1)!.changedActionFraction === 0 ? changed.at(-1)?.depth ?? 1 : null,
      finalConvergence: convergence.at(-1),
      h3ToFinal: { changed: different.length, changedFraction: different.length / grid.length,
        maximumExposureChange: Math.max(0, ...different.map(r => Math.abs(r.difference))) },
      visitedH3ToFinal: { decisions: visited.length, changed: visitedDifferent.length,
        changedFraction: visitedDifferent.length / visited.length,
        maximumExposureChange: Math.max(0, ...visitedDifferent.map(r => Math.abs(r.h3ToFinalDifference))),
        exactH3MaximumDifference: Math.max(0, ...visited.map(r => Math.abs(r.exactH3ToFinalDifference))) },
      cashActions: Array.from({ length: base.model.kernels.length }, (_, leaf) => ({ leaf,
        actions: Array.from({ length: maxDepth }, (_, d) => decideEvent(policy, leaf,
          { equity: base.equities[2], price: base.prices[1], exposure: 0 }, d + 1).exposure) })),
      convergence, elapsedSec: (performance.now() - begin) / 1000 };
    save(`${h3.window.id}-steps-${steps}.json`, { ...result, grid, visited });
    results.push(result);
    console.log(JSON.stringify({ window: h3.window.id, actionSteps: steps, lastActionChangeDepth: result.lastActionChangeDepth,
      finalConvergence: result.finalConvergence, h3ToFinal: result.h3ToFinal, visitedH3ToFinal: result.visitedH3ToFinal,
      elapsedSec: result.elapsedSec }));
  }
}
const bySteps = actionSteps.map(steps => {
  const rows = results.filter(r => r.actionSteps === steps);
  return { actionSteps: steps, windows: rows.length,
    convergedActionAtMaxDepth: rows.filter(r => r.finalConvergence.changedActionFraction === 0).length,
    latestActionChangeDepth: Math.max(...rows.map(r => r.lastActionChangeDepth)),
    maximumFinalValueIncrementSpanBps: Math.max(...rows.map(r => r.finalConvergence.valueIncrementSpan * 10000)),
    gridH3ToFinalChangedFraction: rows.reduce((s, r) => s + r.h3ToFinal.changed, 0) / rows.reduce((s, r) => s + r.gridStates, 0),
    visitedH3ToFinalChangedFraction: rows.reduce((s, r) => s + r.visitedH3ToFinal.changed, 0)
      / rows.reduce((s, r) => s + r.visitedH3ToFinal.decisions, 0),
    maximumVisitedH3ToFinalExposureChange: Math.max(...rows.map(r => r.visitedH3ToFinal.maximumExposureChange)),
    maximumExactH3ToFinalExposureDifference: Math.max(...rows.map(r => r.visitedH3ToFinal.exactH3MaximumDifference)) };
});
const summary = { windows: selected.length, maxDepth, actionSteps, bySteps, elapsedSec: (performance.now() - started) / 1000, results };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, results: undefined }));
