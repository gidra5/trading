/** Check positive quadrature against saved complete training priors. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { compressEventDistribution, eventQuadratureFeatures } from "../packages/bot-algo/src/event-quadrature.js";
import { eventMoveLabel } from "../packages/bot-algo/src/event-distribution.js";
import { eventHolding } from "../packages/bot-algo/src/event-log-policy.js";
const root = path.resolve(__dirname, "..");
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("audit") || !arg("output")) throw new Error("Specify complete-prior audit and new output");
const audit = path.resolve(root, "data/benchmarks", arg("audit")), output = path.resolve(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const config = read(audit, "config.json"), phases = read(audit, "summary.json").map((r: any) => r.phase);
if (config.contract !== "event-prior-audit-v1") throw new Error("Requires full-prior audit");
const costs = read(config.source, "config.json").costs;
const hash = createHash("sha256").update(fs.readFileSync(path.join(audit, "config.json")));
for (const phase of phases) hash.update(fs.readFileSync(path.join(audit, `${phase.id}-models.json`)));
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ ...config, audit, auditHash: hash.digest("hex"),
  quadrature: "Positive support reduction of the entire exact kernel, independently per actual and reciprocal event class / successor. Pin adverse excursion/duration Pareto frontiers; preserve seven moments, falling back to full support on numerical failure.",
  selection: "Numerical approximation check only; complete training-law reference is not an oracle or fitted evaluation outcome" }, null, 2));
const files = ["scripts/compress-event-prior.ts", "packages/bot-algo/src/event-quadrature.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-distribution.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [], started = performance.now();
for (const phase of phases) {
  const input = read(audit, `${phase.id}-models.json`), exact = input.models["hybrid-exact"], t = performance.now();
  const compact = compressEventDistribution(exact);
  let maximumMomentError = 0, maximumConditionalMassError = 0, maximumHoldingErrorBps = 0;
  for (let leaf = 0; leaf < exact.kernels.length; leaf++) {
    const a = exact.kernels[leaf], b = compact.kernels[leaf];
    for (let j = 0; j < eventQuadratureFeatures(a[0]).length; j++) {
      const av = a.reduce((s: number, a: any) => s + a.probability * eventQuadratureFeatures(a)[j], 0);
      const bv = b.reduce((s: number, a: any) => s + a.probability * eventQuadratureFeatures(a)[j], 0);
      maximumMomentError = Math.max(maximumMomentError, Math.abs(av - bv) / Math.max(1, Math.abs(av)));
    }
    const masses = (k: typeof a) => { const m = new Map<string, number>(); for (const a of k) {
      const key = `${a.next}:${eventMoveLabel(a.return, a.duration, exact.clock)}`;
      m.set(key, (m.get(key) ?? 0) + a.probability);
    } return m; };
    const ma = masses(a), mb = masses(b);
    for (const [key, mass] of ma) maximumConditionalMassError = Math.max(maximumConditionalMassError, Math.abs(mass - (mb.get(key) ?? 0)));
    for (const exposure of [-1, -0.5, 0.5, 1]) {
      const value = (k: typeof a) => k.reduce((s: number, a: any) => { const held = eventHolding(exposure, a, costs);
        return s + a.probability * (held.liquidated ? -Infinity : Math.log(held.factor)); }, 0);
      const av = value(a), bv = value(b);
      if (Number.isFinite(av) !== Number.isFinite(bv)) throw new Error("Quadrature changed ruin support");
      if (Number.isFinite(av)) maximumHoldingErrorBps = Math.max(maximumHoldingErrorBps, Math.abs(av - bv) * 1e4);
    }
    if (Math.min(...a.map((a: any) => a.low)) !== Math.min(...b.map((a: any) => a.low))
      || Math.max(...a.map((a: any) => a.high)) !== Math.max(...b.map((a: any) => a.high))) throw new Error("Quadrature lost worst excursion");
  }
  if (maximumMomentError > 1e-9 || maximumConditionalMassError > 1e-10) throw new Error("Quadrature failed numerical contract");
  fs.writeFileSync(path.join(output, `${phase.id}-models.json`), JSON.stringify({ phase, models: {
    "hybrid-control": input.models["hybrid-control"], "hybrid-quadrature": compact,
  } }));
  const result = { phase, exactAtoms: exact.kernels.reduce((s: number, k: any[]) => s + k.length, 0),
    compactAtoms: compact.kernels.reduce((s, k) => s + k.length, 0), maximumMomentError, maximumConditionalMassError, maximumHoldingErrorBps,
    elapsedSec: (performance.now() - t) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2)); console.log(JSON.stringify(result));
}
fs.writeFileSync(path.join(output, "timing.json"), JSON.stringify({ elapsedSec: (performance.now() - started) / 1000 }));
