/** Compare a global continuous relaxation with independently bounded H3 actions. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { restoreEventPolicy, eventTrade } from "../packages/bot-algo/src/event-log-policy.js";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { prepareEventTwoStep } from "../packages/bot-algo/src/event-two-step.js";
import { prepareEventMultiStepUpper } from "../packages/bot-algo/src/event-multi-step-upper.js";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify saved H3 action-probe --source and new --output");
const root = path.resolve(__dirname, ".."), source = path.join(root, "data/benchmarks", arg("source"));
const output = path.join(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const config = read(source, "config.json"); assert.equal(config.contract, "event-three-step-action-probe-v1");
const filename = `${config.window.id}-model.json`, bytes = fs.readFileSync(path.join(config.forecastSource, filename));
assert.equal(createHash("sha256").update(bytes).digest("hex"), config.modelHash);
const policy = restoreEventPolicy(JSON.parse(bytes.toString()).policy), points = Number(arg("points") || 9);
const method = (arg("method") || "uniform") as "uniform" | "marginal";
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-recursive-upper-probe-v1", source, forecastSource: config.forecastSource, modelHash: config.modelHash,
  leaf: config.leaf, account: config.account, shadowPoints: points, method, depth: 3,
  method: "Compile finite-horizon shadow-price upper envelopes from the unchanged saved joint law. Relax orders and maintenance only in the bound; use a conservative reachable-notional proof to exclude exceptional above-cap recovery actions before comparing with the actual discrete model. Compare global root bounds and every available saved H2 continuation lower value. A suggested exposure is a seed, not an optimal action certificate." });
save("sources.json", Object.fromEntries(["scripts/probe-event-multi-step-upper.ts", "packages/bot-algo/src/event-multi-step-upper.ts",
  "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-one-step-upper.ts",
  "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-holding-law.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const started = performance.now(), upper = prepareEventMultiStepUpper(policy.model, policy.costs, "marked", { depth: 3, shadowPoints: points, method });
const preparationSec = (performance.now() - started) / 1000;
save("coefficients.json", upper.coefficients);
const h1 = decideEventOneStep(policy.model.kernels[config.leaf], config.account, policy.costs, "marked");
const h2 = prepareEventTwoStep(policy.model, policy.costs, "marked")(config.leaf, config.account, { maxEvaluations: 64 });
const recorded = fs.existsSync(path.join(source, "summary.json")) ? read(source, "summary.json").results : [];
const lowers = [h1.value, h2.lowerValue, Math.max(-Infinity, ...recorded.filter((r: any) => r.result.complete && Number.isFinite(r.result.lowerValue))
  .map((r: any) => r.result.lowerValue))];
const results = [1, 2, 3].map(h => {
  const bound = upper.query(config.leaf, config.account, h), lower = lowers[h - 1];
  assert.ok(bound.upperValue >= lower - 1e-9, `Global H${h} bound below a feasible value`);
  return { depth: h, ...bound, lowerValue: lower, gapBps: (bound.upperValue - lower) * 10000,
    seedQuantities: bound.seedExposures.map(x => eventTrade(config.account,
      Math.max(-policy.costs.maxLeverage, Math.min(policy.costs.maxLeverage, x)), policy.costs)?.quantity ?? 0) };
});
const continuation: any[] = [];
for (let i = 0; i < recorded.length; i++) {
  const file = `action-${i}-branches.json`; if (!fs.existsSync(path.join(source, file))) continue;
  const rows = read(source, file).map((r: any) => {
    const bound = upper.query(r.next, r.account, 2);
    assert.ok(bound.upperValue >= r.lowerValue - 1e-9, "Recursive bound below saved continuation");
    return { probability: r.probability, finite: Number.isFinite(bound.upperValue), excessBps: (bound.upperValue - r.lowerValue) * 10000 };
  });
  continuation.push({ quantity: recorded[i].result.quantity, count: rows.length, finite: rows.filter((r: any) => r.finite).length,
    meanExcessBps: rows.reduce((s: number, r: any) => s + r.probability * r.excessBps, 0), maximumExcessBps: Math.max(...rows.map((r: any) => r.excessBps)) });
}
const directional = [-1, 1].map(side => ({ side,
  ...upper.query(config.leaf, config.account, 3, side as -1 | 1) }));
const summary = { preparationSec, results, directional, continuation, elapsedSec: (performance.now() - started) / 1000 };
save("summary.json", summary); console.log(JSON.stringify(summary));
