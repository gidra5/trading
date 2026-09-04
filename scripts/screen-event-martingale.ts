/** Test whether event-sign skill also appears on a zero-drift price process. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import type { EventCandle, MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { predictEventSign, trainEventSign } from "../packages/bot-algo/src/event-sign.js";
import { eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

/** Each close changes by +/-tanh(|historical log return|), with independent
 * equiprobable signs. Conditional on the entire exogenous amplitude schedule,
 * E[P_(t+1)|past] = P_t exactly. This avoids the Jensen drift of log-sign flips.
 * Wicks are only synthetic envelopes, not reconstructed intraminute paths. */
export function eventMartingaleCandles(c: readonly EventCandle[], seed: number): EventCandle[] {
  if (!c.length || !Number.isInteger(seed) || seed <= 0 || seed > 0xffffffff) throw new Error("Invalid martingale source or seed");
  let state = seed >>> 0, price = c[0].close;
  const result: EventCandle[] = [{ ...c[0], secondBasis: undefined }];
  for (let i = 1; i < c.length; i++) {
    if (c[i].openTime - c[i - 1].openTime !== 60_000) throw new Error("Gap in martingale amplitude schedule");
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    const sign = state < 0x80000000 ? -1 : 1;
    const logReturn = Math.log(c[i].close / c[i - 1].close), magnitude = Math.tanh(Math.abs(logReturn));
    const close = price * (1 + sign * magnitude);
    const highWick = Math.max(0, Math.log(c[i].high / Math.max(c[i].open, c[i].close, c[i - 1].close)));
    const lowWick = Math.max(0, Math.log(Math.min(c[i].open, c[i].close, c[i - 1].close) / c[i].low));
    const reflected = sign !== Math.sign(logReturn);
    const candle = { openTime: c[i].openTime, open: price, close,
      high: Math.max(price, close) * Math.exp(reflected ? lowWick : highWick),
      low: Math.min(price, close) * Math.exp(-(reflected ? highWick : lowWick)), volume: c[i].volume };
    if (Object.values(candle).some(v => !Number.isFinite(v)) || !(candle.low > 0)) throw new Error("Invalid synthetic candle");
    result.push(candle); price = close;
  }
  return result;
}

export async function main() {
  const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
  const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
  if (!arg("source") || !arg("output")) throw new Error("Specify joint-policy source and new output");
  const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
  if (fs.existsSync(output)) throw new Error("Choose a new output directory");
  const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
  const jc = read(source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
  if (jc.contract !== "event-volatility-law-policy-v1") throw new Error("Requires joint event source");
  const phases = eventRefitOrigins(jc.window.startTime, oc.foldCount, oc.foldDays), penalty = 0.1, seeds = [11, 29, 47, 83];
  const fits = phases.map(p => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
  const actual = loadEventCandles(fits[0].trainStart - DAY, phases.at(-1)!.endTime + DAY);
  const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(JSON.stringify(actual));
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-martingale-screen-v1", source, phases, penalty, seeds,
    sourceHash: hash.digest("hex"), inputs: "Original 20 run features plus three fast volatility features; no external price/flow information.",
    null: "Independent symmetric arithmetic close returns with magnitude tanh(abs(historical log return)); actual timestamps and volume schedule, reflected candle wicks; fresh causal event labels on each path.",
    caveat: "Four deterministic pseudo-random seeds are a diagnostic, not a formal significance test. Null and actual event populations differ because clocks are regenerated. No final inspector outcomes, hyperparameter selection or economic-policy claims." }, null, 2));
  const files = ["scripts/screen-event-martingale.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-sign.ts"];
  fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
  const results = [], started = performance.now();
  for (const seed of [null, ...seeds]) {
    const c = seed === null ? actual : eventMartingaleCandles(actual, seed);
    for (const [i, phase] of phases.entries()) {
      const train = makeSamples(c, sc.clock, fits[i].trainStart, fits[i].trainEnd, [], sc.stride, "chain", sc.featureNames);
      const test = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames);
      if (train.length < 100 || test.length < 100 || train.some(r => c[r.end].openTime + 60000 >= phase.startTime)) throw new Error("Insufficient or future training targets");
      const features = (r: MoveSample) => [...r.features, ...eventFastVolatilityFeatures(c, r.start)];
      const head = trainEventSign(train.map(r => ({ features: features(r), return: r.return })), penalty);
      fs.writeFileSync(path.join(output, `${seed ?? "actual"}-${phase.id}-head.json`), JSON.stringify(head));
      const predictions = test.filter(r => r.return !== 0).map(r => ({ time: c[r.start].openTime + 60000, availableAt: c[r.end].openTime + 60000,
        probability: predictEventSign(head, features(r)), return: r.return, duration: r.duration }));
      const count = predictions.length, mean = (f: (r: typeof predictions[number]) => number) => predictions.reduce((s, r) => s + f(r), 0) / count;
      const correct = (r: typeof predictions[number]) => (r.probability >= 0.5) === (r.return > 0);
      const totalMagnitude = predictions.reduce((s, r) => s + Math.abs(r.return), 0);
      const result = { seed, phase: phase.id, training: train.length, events: count,
        signLoss: mean(r => -Math.log(r.return > 0 ? r.probability : 1 - r.probability)),
        accuracy: mean(r => Number(correct(r))), meanReturnBps: mean(r => r.return * 1e4),
        predictedSignGrossBps: mean(r => Math.sign(r.probability - 0.5) * r.return * 1e4),
        magnitudeWeightedAccuracy: predictions.reduce((s, r) => s + Math.abs(r.return) * Number(correct(r)), 0) / totalMagnitude,
        averageCorrectMagnitudeBps: predictions.filter(correct).reduce((s, r) => s + Math.abs(r.return) * 1e4, 0) / predictions.filter(correct).length,
        averageWrongMagnitudeBps: predictions.filter(r => !correct(r)).reduce((s, r) => s + Math.abs(r.return) * 1e4, 0) / predictions.filter(r => !correct(r)).length };
      results.push(result);
      fs.writeFileSync(path.join(output, `${seed ?? "actual"}-${phase.id}-predictions.json`), JSON.stringify(predictions));
      fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
      console.log(JSON.stringify(result));
    }
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === __filename) void main();
