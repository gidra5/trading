/** Paired forecast errors and cost-covering forecast tails; no portfolio claims. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify hold-path screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json");
if (config.contract !== "event-hold-path-screen-v1") throw new Error("Requires hold-path forecast source");
const mean = (v: number[]) => v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
const meanRequired = (v: number[]) => { assert.ok(v.length); return mean(v)!; };
const seed = 20260904, replicates = 2000;
const phases = config.phases.map((phase: any) => {
  const rows = read(source, `${phase.id}-predictions.json`), parent = read(config.source, "config.json");
  const base = read(parent.source, `${phase.id}-policy.json`), costs = base.costs;
  const entryLog = -Math.log(1 + costs.maxLeverage * (costs.feeBps + costs.slippageBps) / 10000);
  assert.ok(rows.every((r: any) => r.time >= phase.startTime && r.horizons.every((h: any) => h.availableAt < phase.endTime)));
  return { phase, rows, entryLog };
});
fs.mkdirSync(output, { recursive: true });
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const { phase } of phases) hash.update(fs.readFileSync(path.join(source, `${phase.id}-predictions.json`)));
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-hold-path-audit-v1", source, sourceHash: hash.digest("hex"), seed, replicates,
  method: "Compare ridge holding and continuation-increment MSE against training-only clock-state means on paired UTC decision days; positive gains favor ridge. For each declared horizon, select the higher predicted full-long/full-short holding value only when it covers the analytical proportional entry cost. Report all chosen probes, including cap violations and overlapping paths, rather than silently removing them.",
  caveat: "No final outcomes, fitting or policy selection. Descriptive resampling of reused dependent days. Cost-covering probes are not an executable account or a portfolio backtest. Entry ignores lot/min/max constraints; complete-prefix settlement follows the holding-value convention." }, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename));
const paired = [];
for (const horizon of [1, 2, 3]) for (const target of horizon === 1 ? ["holding"] : ["holding", "increment"]) {
  const origins = phases.map(({ phase, rows }: any) => {
    const days = new Map<number, { day: number; sum: number; count: number }>();
    for (const row of rows) {
      const h = row.horizons[horizon - 1], first = row.horizons[0], day = Math.floor(row.time / DAY), d = days.get(day) ?? { day, sum: 0, count: 0 };
      for (const side of [0, 1]) {
        const actual = h.actual[side] - (target === "increment" ? first.actual[side] : 0);
        const predicted = h.predicted[side] - (target === "increment" ? first.predicted[side] : 0);
        const clock = h.clock[side] - (target === "increment" ? first.clock[side] : 0);
        d.sum += ((actual - clock) ** 2 - (actual - predicted) ** 2) * 1e8; d.count++;
      }
      days.set(day, d);
    }
    return { phase: phase.id, days: [...days.values()] };
  });
  let state = seed >>> 0;
  const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
  const samples = Array.from({ length: replicates }, () => meanRequired(origins.map(o => {
    let sum = 0, count = 0;
    for (let i = 0; i < o.days.length; i++) { const d = o.days[Math.floor(random() * o.days.length)]; sum += d.sum; count += d.count; }
    return sum / count;
  }))).sort((a, b) => a - b);
  const gains = origins.map(o => o.days.reduce((s, d) => s + d.sum, 0) / o.days.reduce((s, d) => s + d.count, 0));
  paired.push({ horizon, target, originGainsBpsSquared: gains, meanGainBpsSquared: meanRequired(gains),
    interval95: [samples[Math.floor(replicates * .025)], samples[Math.floor(replicates * .975)]] });
}
const tails = phases.map(({ phase, rows, entryLog }: any) => ({ phase, rows: [1, 2, 3].map(horizon => {
  const selected = rows.flatMap((r: any) => {
    const h = r.horizons[horizon - 1], side = h.predicted[1] > h.predicted[0] ? 1 : 0;
    return h.predicted[side] + entryLog > 0 ? [{ time: r.time, availableAt: h.availableAt, side: side ? "long" : "short",
      predictedNetBps: (entryLog + h.predicted[side]) * 10000, actualNetBps: (entryLog + h.actual[side]) * 10000,
      capBreaches: h.capBreaches[side] }] : [];
  });
  const days = [...new Set(selected.map((r: any) => Math.floor(r.time / DAY)))];
  fs.writeFileSync(path.join(output, `${phase.id}-h${horizon}-selected.json`), JSON.stringify(selected));
  return { horizon, count: selected.length, distinctDays: days.length, long: selected.filter((r: any) => r.side === "long").length,
    capBreaches: selected.filter((r: any) => r.capBreaches > 0).length,
    meanPredictedNetBps: mean(selected.map((r: any) => r.predictedNetBps)), meanActualNetBps: mean(selected.map((r: any) => r.actualNetBps)),
    positiveFraction: mean(selected.map((r: any) => Number(r.actualNetBps > 0))) };
}) }));
const result = { paired, tails };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result));
