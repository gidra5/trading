/** Profile certified two-event search on saved forecasts before scaling replay. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { decideEvent, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { decideEventTwoStep } from "../packages/bot-algo/src/event-two-step.js";
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify saved one-step screen --source and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("source")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, file: string) => JSON.parse(fs.readFileSync(path.join(d, file), "utf8"));
const config = read(source, "config.json"); assert.equal(config.contract, "event-one-step-screen-v1");
const phase = arg("phase") ? config.phases.find((p: any) => p.id === arg("phase")) : config.phases[0];
assert.ok(phase, "Unknown saved phase");
const filename = path.join(config.source, `${phase.id}-policy.json`), policy = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
const selection = arg("selection") || "initial-inventory";
assert.ok(["initial-inventory", "marked-orders"].includes(selection));
const probes = selection === "initial-inventory"
  ? read(source, `${phase.id}-probes.json`).filter((r: any, i: number) => !i || r.kind === "initial-inventory")
  : read(source, `${phase.id}-marked-trades.json`).filter((r: any) => r.orderQuantity).map((r: any) => ({
    kind: "marked-order", time: r.time, leaf: r.leaf, account: { equity: r.equityBefore, price: r.order.price, exposure: r.exposureBefore },
    markedQuantity: r.order.quantity, markedValue: r.order.value }));
const count = Number(arg("count") || 1), offset = Number(arg("offset") || 0);
const budgets = (arg("budgets") || "8").split(",").map(Number), tolerance = 1e-7;
assert.ok(Number.isInteger(offset) && offset >= 0 && Number.isInteger(count) && count > 0 && offset + count <= probes.length
  && budgets.every(b => Number.isInteger(b) && b >= 2));
const selected = probes.slice(offset, offset + count);
const modelHash = createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-two-step-saved-probe-v1", source, forecastSource: config.source, phase, modelHash,
  budgets, tolerance, selection, offset, probes: selected, terminal: "marked",
  method: "Same frozen joint kernel and same observed account. Exact one-event continuation; root seeds are cash/hold, exact one-step and old depth-two grid action. Profile a bounded search and retain unresolved upper/lower gaps. No candle replay, new fit, policy promotion, or final-window selection." });
save("sources.json", Object.fromEntries(["scripts/probe-event-two-step.ts", "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-holding-law.ts",
  "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-log-policy.ts"].map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results = [], started = performance.now();
for (const probe of selected) for (const maxEvaluations of budgets) {
  const grid = decideEvent(policy, probe.leaf, probe.account, 2), begin = performance.now();
  const result = decideEventTwoStep(policy.model, probe.leaf, probe.account, policy.costs, "marked",
    { tolerance, maxEvaluations, seedQuantities: [grid.quantity] });
  const row = { kind: probe.kind, time: probe.time, account: probe.account, leaf: probe.leaf, maxEvaluations,
    markedQuantity: probe.markedQuantity, markedValue: probe.markedValue,
    rootAtoms: policy.model.kernels[probe.leaf].length, gridQuantity: grid.quantity, result,
    gapBps: result.gap * 10000, improvedSeedBps: (result.value - result.initial.value) * 10000, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(row); save("summary.json", { results, elapsedSec: (performance.now() - started) / 1000 }); console.log(JSON.stringify(row));
}
