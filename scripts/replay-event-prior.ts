/** Isolate training-prior compression at an already chosen active depth. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { buildEventPolicy, restoreEventPolicy, serializeEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventOriginScore, eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("audit") || !arg("policy") || !arg("output")) throw new Error("Specify prior audit, saved policy replay and new output");
const audit = path.resolve(root, "data/benchmarks", arg("audit")), source = path.resolve(root, "data/benchmarks", arg("policy"));
const output = path.resolve(root, "data/benchmarks", arg("output")), modelName = arg("model") || "hybrid-exact";
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const ac = read(audit, "config.json"), config = read(source, "config.json"), original = read(config.source, "config.json"), sc = read(original.source, "config.json");
if (ac.contract !== "event-prior-audit-v1" || config.contract !== "event-volatility-law-policy-v1"
  || ac.forecast !== config.forecast || ac.source !== config.source) throw new Error("Mismatched paired sources");
const choice = read(source, "selection.json").ranking.find((r: any) => r.choice === "joint-volatility" && r.trades > 0);
if (!choice) throw new Error("No saved active hybrid depth");
const window = config.window, phases = [...eventRefitOrigins(window.startTime, original.foldCount, original.foldDays), { ...window, id: "final" }];
fs.mkdirSync(output, { recursive: true });
const hash = createHash("sha256").update(fs.readFileSync(path.join(audit, "config.json"))).update(fs.readFileSync(path.join(source, "selection.json")));
for (const phase of phases) hash.update(fs.readFileSync(path.join(audit, `${phase.id}-models.json`))).update(fs.readFileSync(path.join(source, `${phase.id}-policy.json`)));
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-prior-policy-audit-v1", audit, source, sourceHash: hash.digest("hex"), modelName, choice, window,
  caveat: "Fixed active depth chosen on prior origins by the compressed incumbent. Diagnostic active replay disables cash gate. No tuning or promotion from final returns." }, null, 2));
const files = ["scripts/replay-event-prior.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-distribution.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const c = loadEventCandles(phases[0].startTime - 2 * DAY, window.endTime + DAY), rows: any[] = [], started = performance.now();
for (const phase of phases) {
  if (phase.id === "final") {
    const ranking = ["control", "exact"].map(name => ({ choice: name, ...eventOriginScore(rows.map(r => r[name]), sc.riskPenalty) })).sort((a, b) => b.score - a.score);
    fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ ranking, cash: ranking[0].score <= 0 }, null, 2));
  }
  const t = performance.now(), controlPolicy = restoreEventPolicy(read(source, `${phase.id}-policy.json`));
  const models = read(audit, `${phase.id}-models.json`).models;
  if (JSON.stringify(controlPolicy.model) !== JSON.stringify(models["hybrid-control"]) || !models[modelName]) throw new Error("Saved control model changed or missing alternative");
  const p = buildEventPolicy(models[modelName], controlPolicy.costs, { depths: choice.depth,
    referenceEquity: controlPolicy.equities[2], referencePrice: controlPolicy.prices[1], actionSteps: sc.actionSteps });
  fs.writeFileSync(path.join(output, `${phase.id}-policy.json`), JSON.stringify(serializeEventPolicy(p)));
  const control = replayEventPolicy(c, controlPolicy, phase.startTime, phase.endTime, choice.depth, { trace: true });
  const exact = replayEventPolicy(c, p, phase.startTime, phase.endTime, choice.depth, { trace: true });
  if (phase.id !== "final") {
    const expected = read(source, `${phase.id}-scores.json`).rows.find((r: any) => r.choice === "joint-volatility" && r.depth === choice.depth);
    if (expected.returnPct !== control.returnPct || expected.trades !== control.trades || expected.maxDrawdownPct !== control.maxDrawdownPct) throw new Error("Incumbent replay changed");
  }
  fs.writeFileSync(path.join(output, `${phase.id}-trades.json`), JSON.stringify(exact.trace));
  const { trace: tc, ...cm } = control, { trace: te, ...em } = exact;
  if (tc.length !== te.length || tc.some((r: any, i) => r.time !== (te[i] as any).time || r.leaf !== (te[i] as any).leaf)) throw new Error("Prior changed decision-state clock");
  const result = { phase, control: cm, exact: em,
    changedOrderDecisions: tc.filter((r: any, i) => Math.abs(r.orderQuantity - (te[i] as any).orderQuantity) > 1e-12).length,
    elapsedSec: (performance.now() - t) / 1000 };
  rows.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ rows, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
  console.log(JSON.stringify({ phase: phase.id, depth: choice.depth, control: { returnPct: cm.returnPct, drawdown: cm.maxDrawdownPct, trades: cm.trades },
    exact: { returnPct: em.returnPct, drawdown: em.maxDrawdownPct, trades: em.trades }, changedOrderDecisions: result.changedOrderDecisions, elapsedSec: result.elapsedSec }));
}
