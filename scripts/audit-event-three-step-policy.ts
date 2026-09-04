/** Audit the complete H3 policy's feasible values and global certificates. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { prepareEventThreeStep } from "../packages/bot-algo/src/event-three-step.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify a saved recursive-upper audit --source and new --output");
const root = path.resolve(__dirname, ".."), source = path.join(root, "data/benchmarks", arg("source"));
const output = path.join(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
assert.equal(config.contract, "event-recursive-upper-audit-v1");
const cases = config.cases.slice(0, Number(arg("cases") || 100)), budgets = [1, 4], tolerance = 1e-7;
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-three-step-policy-audit-v1", source, cases, budgets, tolerance,
  method: "Independently enumerate all H3 root and future lots on saved small laws. Check the feasible lower policy, every candidate upper bound and the global upper certificate, with one or four expensive root evaluations. A winning candidate without a small global gap must remain unresolved." });
save("sources.json", Object.fromEntries(["scripts/audit-event-three-step-policy.ts", "packages/bot-algo/src/event-three-step.ts",
  "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-one-step.ts",
  "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-holding-law.ts",
  "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/test/event-bellman-reference.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const rows = [], start = performance.now();
for (const c of cases) for (const terminal of ["marked", "friction"] as const) {
  const reference = eventBellmanReference(c.model, c.costs, { terminal, maxNodes: 1000000 });
  const decide = prepareEventThreeStep(c.model, c.costs, terminal, { shadowPoints: 9 });
  for (const exposure of [0, -1.9, 1.9]) {
    const account = { equity: 100, price: 10, exposure }, optimal = reference.decide(0, account, 3);
    for (const maxRootEvaluations of budgets) {
      const actual = decide(0, account, { maxEvaluations: 512, maxRootEvaluations, tolerance });
      const chosenValue = reference.actionValue(0, account, 3, actual.quantity);
      const le = (a: number, b: number) => a === b || a <= b + 1e-9;
      const candidatesValid = actual.candidates.every(candidate => le(reference.actionValue(0, account, 3, candidate.quantity), candidate.upperValue));
      const valid = le(actual.lowerValue, chosenValue) && le(optimal.value, actual.upperValue) && candidatesValid
        && (!actual.converged || actual.feasible && actual.gap <= tolerance && le(optimal.value - tolerance, actual.lowerValue));
      const row = { id: c.id, terminal, account, maxRootEvaluations, optimal, chosenValue, actual, valid };
      rows.push(row);
      if (!valid) { save("failure.json", { case: c, row }); throw new Error(`H3 policy audit failed: ${c.id} ${terminal} ${exposure}`); }
    }
  }
}
const byBudget = budgets.map(budget => {
  const selected = rows.filter(r => r.maxRootEvaluations === budget), certified = selected.filter(r => r.actual.converged);
  return { budget, queries: selected.length, certified: certified.length,
    maximumCertifiedRegretBps: Math.max(0, ...certified.map(r => r.optimal.value === r.chosenValue ? 0 : (r.optimal.value - r.chosenValue) * 10000)),
    finiteUpper: selected.filter(r => Number.isFinite(r.actual.upperValue)).length };
});
const summary = { cases: cases.length, queries: rows.length, byBudget, elapsedSec: (performance.now() - start) / 1000, rows };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, rows: undefined }));
