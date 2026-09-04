/** Compare the local-law forecast with causal direction baselines and paired days. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures } from "../packages/bot-algo/src/event-distribution.js";
import { loadEventCandles } from "./research-event-policy.js";
const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify local-value screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), screen = read(config.source, "config.json"), jc = read(screen.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const selected = read(source, "summary.json").ranking[0].count;
const c = loadEventCandles(config.phases[0].startTime - 2 * DAY, config.phases.at(-1).endTime), byTime = new Map(c.map((r, i) => [r.openTime + 60000, i]));
const runIndex = sc.featureNames.indexOf("runSign"), dcIndex = sc.featureNames.indexOf("dcDirection");
assert.ok(runIndex >= 0 && dcIndex >= 0);
const mean = (values: number[]) => values.reduce((s, v) => s + v, 0) / values.length;
const mse = (predicted: number[], actual: number[]) => mean(predicted.map((v, i) => (v - actual[i]) ** 2));
const seed = 20260904; let state = seed;
const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
const phases: any[] = [], dayBlocks: Array<Array<{ sum: number; count: number }>> = [];
for (const phase of config.phases) {
  const model = read(source, `${phase.id}-model.json`), rows = read(source, `${phase.id}-predictions.json`);
  const up = model.targets.reduce((s: number, r: number[]) => s + r[2], 0), nonflat = model.targets.reduce((s: number, r: number[]) => s + r[3], 0);
  const prior = nonflat ? up / nonflat : 0.5;
  const groups = new Map<number, { count: number; sums: number[] }>();
  for (let i = 0; i < model.design.length; i++) {
    const raw = model.design[i][dcIndex] * model.scales[dcIndex] + model.means[dcIndex], direction = Math.round(raw);
    assert.ok(Math.abs(raw - direction) < 1e-8 && Math.abs(direction) <= 1);
    const group = groups.get(direction) ?? { count: 0, sums: model.targets[i].map(() => 0) }; group.count++;
    model.targets[i].forEach((v: number, j: number) => group.sums[j] += v); groups.set(direction, group);
  }
  const observed = rows.map((r: any) => {
    const features = eventFeatures(c, byTime.get(r.time)!, sc.featureNames, sc.clock);
    const group = groups.get(features[dcIndex]); assert.ok(group);
    return { ...r, runSign: features[runIndex], dcDirection: features[dcIndex],
      clockProbability: (group.sums[2] + 1) / (group.sums[3] + 2), clockHolding: group.sums.slice(0, 2).map(v => v / group.count) };
  });
  const nonflatRows = observed.filter((r: any) => r.return !== 0);
  const accuracy = (fn: (r: any) => number) => mean(nonflatRows.map((r: any) => Number((fn(r) > 0) === (r.return > 0))));
  const local = (r: any) => r.local.find((v: any) => v.count === selected);
  const days = new Map<string, { sum: number; count: number }>();
  for (const r of rows) {
    const day = new Date(r.time).toISOString().slice(0, 10), block = days.get(day) ?? { sum: 0, count: 0 };
    block.sum += mse(local(r).holding, r.actual) - mse(r.ridge, r.actual); block.count++; days.set(day, block);
  }
  dayBlocks.push([...days.values()]);
  phases.push({ phase, count: rows.length, nonflat: nonflatRows.length, selectedNeighbors: selected, trainingUpProbability: prior,
    directionAccuracy: { trainingMajority: accuracy(() => prior - 0.5), currentRawRun: accuracy(r => r.runSign),
      currentDcDirection: accuracy(r => r.dcDirection), reverseRawRun: accuracy(r => -r.runSign), reverseDcDirection: accuracy(r => -r.dcDirection),
      localSign: accuracy(r => local(r).upProbability - 0.5),
      ridgeHolding: accuracy(r => r.ridge[1] - r.ridge[0]), localHolding: accuracy(r => local(r).holding[1] - local(r).holding[0]) },
    localSignAgreementWithReverseDc: mean(nonflatRows.map((r: any) => Number((local(r).upProbability > 0.5) === (-r.dcDirection > 0)))),
    localSignAgreementWithReverseRaw: mean(nonflatRows.map((r: any) => Number((local(r).upProbability > 0.5) === (-r.runSign > 0)))),
    trainingClockGroups: [...groups].map(([direction, g]) => ({ direction, count: g.count, upProbability: (g.sums[2] + 1) / (g.sums[3] + 2), holding: g.sums.slice(0, 2).map(v => v / g.count) })),
    brier: { prior: mean(nonflatRows.map((r: any) => (prior - Number(r.return > 0)) ** 2)),
      clock: mean(nonflatRows.map((r: any) => (r.clockProbability - Number(r.return > 0)) ** 2)),
      local: mean(nonflatRows.map((r: any) => (local(r).upProbability - Number(r.return > 0)) ** 2)) },
    clockHoldingMse: mean(observed.map((r: any) => mse(r.clockHolding, r.actual))),
    differenceMseBpsSquared: mean(rows.map((r: any) => mse(local(r).holding, r.actual) - mse(r.ridge, r.actual))) * 1e8,
    days: [...days].map(([date, value]) => ({ date, ...value })) });
}
const bootstrap = Array.from({ length: 2000 }, () => mean(dayBlocks.map(blocks => {
  let sum = 0, count = 0;
  for (let i = 0; i < blocks.length; i++) { const b = blocks[Math.floor(random() * blocks.length)]; sum += b.sum; count += b.count; }
  return sum / count;
})) * 1e8).sort((a, b) => a - b);
const summary = { selectedNeighbors: selected, phases, meanDifferenceMseBpsSquared: mean(phases.map(p => p.differenceMseBpsSquared)),
  pairedDayBootstrap: { seed, repetitions: 2000, lower95: bootstrap[50], upper95: bootstrap[1949] } };
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, selectedNeighbors: selected,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(fs.readFileSync(path.join(source, "summary.json"))).digest("hex"),
  method: "Compare selected neighborhood sign and holding-value direction with training majority, observed run/DC direction and their opposites. A training-only DC-state grouping supplies Laplace-smoothed sign probabilities and mean holding values. Resample whole decision days jointly for ridge/local value residuals within each prior origin, then average the three origin MSE differences equally. Negative differences favor local averages.",
  caveat: "Exploratory uncertainty after choosing neighbor count on these same research origins. Adjacent days and origins may remain dependent. No independent significance claim, sequential backtest, or final inspector outcome. Zero sign forecasts use the same down tie-break as the forecast screen; exactly flat outcomes are excluded from sign metrics." }, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ ...summary, phases: phases.map(({ days, ...p }) => p) }));
