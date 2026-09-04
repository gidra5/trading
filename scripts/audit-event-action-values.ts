/** Completed-outcome diagnostic. Never fit/select a policy from this output. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, observeMove, type MoveAtom } from "../packages/bot-algo/src/event-distribution.js";
import { buildEventOutcomeLookahead, decideEvent, eventActionValues, eventHolding, restoreEventPolicy,
  type EventAccount, type EventPolicy, type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSizeSignGroup, mixEventSizeSigns, reweightEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCalibrationRanges, loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sourceId = arg("source"), outputId = arg("output");
if (!sourceId || !outputId) throw new Error("Specify source replay and new diagnostic output");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = <T = any>(dir: string, name: string): T => JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
const config = read(source, "config.json");
if (config.contract !== "event-sign-lookahead-replay-v1" || !config.compareBase) throw new Error("Requires explicit saved policy selection");
const summaries = read(source, "summary.json") as Array<{ window: { id: string; startTime: number; endTime: number };
  chosen: { choice: string; depth: number }; cash: boolean; appliedBlend: number; test: { returnPct: number } }>;
const ids = arg("windows").split(",").filter(Boolean);
if (ids.some(id => !summaries.some(s => s.window.id === id))) throw new Error("Unknown window");
fs.mkdirSync(output, { recursive: true });
const files = ["scripts/audit-event-action-values.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-log-policy.ts",
  "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-distribution.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const fingerprint = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(fs.readFileSync(path.join(source, "summary.json")));
const results = [];
const mean = (values: number[]) => values.length ? values.reduce((s, x) => s + x, 0) / values.length : null;

function directValue(p: EventPolicy, account: EventAccount, trade: EventAccount, kernel: readonly MoveAtom[], depth: number) {
  let reward = 0, continuation = 0, terminal = 0;
  for (const atom of kernel) {
    if (!atom.probability) continue;
    const hold = eventHolding(trade.exposure, atom, p.costs);
    if (hold.liquidated) return { reward: -Infinity, continuation: -Infinity, terminal: -Infinity, value: -Infinity };
    reward += atom.probability * Math.log(hold.factor);
    const settlement = Math.log(1 - Math.abs(hold.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4);
    terminal += atom.probability * settlement;
    continuation += atom.probability * (depth > 1 ? decideEvent(p, atom.next, {
      equity: trade.equity * hold.factor, price: trade.price * (1 + atom.return), exposure: hold.exposure,
    }, depth - 1).value : settlement);
  }
  const cost = Math.log(trade.equity / account.equity);
  return { reward: cost + reward, continuation, terminal: cost + reward + terminal, value: cost + reward + continuation };
}

for (const s of summaries) {
  if (ids.length && !ids.includes(s.window.id)) continue;
  const started = performance.now(), id = s.window.id;
  const original = read(config.source, `${id}-model.json`);
  const heads = read(source, `${id}-sign-model.json`);
  if (s.chosen.choice.startsWith("head-lag")) Object.assign(heads, read(source, `${id}-lag-head.json`));
  const projected = ["head", "head-lag"].includes(s.chosen.choice) && config.projectContinuation ? read(source, `${id}-continuation.json`) : original;
  const phases = [];
  const refitComparison: any[] = [];
  for (const phase of ["calibration", "test"] as const) {
    const p = restoreEventPolicy((phase === "test" ? projected.policy : projected.selectionPolicy) as SerializedEventPolicy);
    const head: EventSizeSignHead = phase === "test" ? heads.head : heads.selectionHead;
    const useHead = s.chosen.choice !== "base" && heads.selected;
    if (useHead && !head?.gate) throw new Error("Audit currently requires size/sign head or original policy");
    const lookahead = useHead ? buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, head.thresholdLogBps)) : undefined;
    const ranges = phase === "test" ? [s.window]
      : eventCalibrationRanges(original.policyCalibrationStart, original.calibrationEnd, original.selectionExcludedWindows);
    const c = loadEventCandles(ranges[0].startTime - 2 * DAY, ranges.at(-1)!.endTime + DAY);
    const byTime = new Map(c.map((candle, i) => [candle.openTime + 60_000, i]));
    const rows: any[] = [];
    const groups = Array.from({ length: 5 }, (_, group) => ({ group, samples: 0, predictedMass: 0, realizedMass: 0,
      predictedLogReturnContribution: 0, realizedLogReturnContribution: 0 }));
    let maximumReproductionValueError = 0;
    for (const range of ranges) {
      const replay = replayEventPolicy(c, p, range.startTime, range.endTime, s.chosen.depth, { cash: s.cash, trace: true,
        ...(lookahead ? { sizeSign: { head, blend: s.appliedBlend, lookahead, fastVolatility: heads.selected.fastVolatility, eventHistory: heads.selected.eventHistory } } : {}) });
      if (phase === "test") {
        if (Math.abs(replay.returnPct - s.test.returnPct) > 1e-10) throw new Error("Saved return does not reproduce");
        const savedTrace = read(source, `${id}-trades.json`);
        if (JSON.stringify(savedTrace) !== JSON.stringify(replay.trace)) throw new Error("Saved decisions do not reproduce exactly");
        if (useHead && process.argv.includes("--compare-refits")) {
          for (const modelAge of ["selection", "final"] as const) {
            const alternate = modelAge === "selection" ? restoreEventPolicy(projected.selectionPolicy) : p;
            for (const headAge of ["selection", "final"] as const) {
              const alternateHead: EventSizeSignHead = headAge === "selection" ? heads.selectionHead : heads.head;
              const alternateLookahead = buildEventOutcomeLookahead(alternate, 5, a => eventSizeSignGroup(a.return, alternateHead.thresholdLogBps));
              const result = replayEventPolicy(c, alternate, range.startTime, range.endTime, s.chosen.depth, { cash: s.cash, trace: true,
                sizeSign: { head: alternateHead, blend: s.appliedBlend, lookahead: alternateLookahead,
                  fastVolatility: heads.selected.fastVolatility, eventHistory: heads.selected.eventHistory } });
              const { trace, ...metrics } = result;
              refitComparison.push({ modelAge, headAge, ...metrics, meanExpectedReturnBps: mean(trace.map(t => t.expectedReturnBps as number)) });
              fs.writeFileSync(path.join(output, `${id}-${modelAge}-model-${headAge}-head.json`), JSON.stringify(trace));
            }
          }
        }
      }
      for (const raw of replay.trace) {
        const t = raw as any, i = byTime.get(t.time)!;
        const move = observeMove(c, i, p.model.clock, p.model.featureNames)!;
        if (c[move.end].openTime + 60_000 > range.endTime) continue;
        const account = { equity: t.equityBefore, price: c[i].close, exposure: t.exposureBefore };
        const kernel = lookahead ? reweightEventSizeSigns(p.model.kernels[t.leaf], head.thresholdLogBps, t.sizeSignProbabilities, s.appliedBlend) : p.model.kernels[t.leaf];
        const values = eventActionValues(p, t.leaf, account, s.chosen.depth, lookahead
          ? { lookahead, mass: mixEventSizeSigns(lookahead.masses[t.leaf], t.sizeSignProbabilities, s.appliedBlend) } : undefined);
        const selected = values.find(a => Math.abs(a.quantity - t.order.quantity) < 1e-12)!;
        if (!selected) throw new Error("Saved candidate is unavailable");
        maximumReproductionValueError = Math.max(maximumReproductionValueError, Math.abs(selected.value - t.order.value));
        const chosen = [selected, values.find(a => a.target === 0), values.find(a => a.quantity === 0)]
          .filter((v, j, all) => v && all.indexOf(v) === j).map(trade => {
            const predicted = directValue(p, account, trade!, kernel, s.chosen.depth);
            const realized = directValue(p, account, trade!, [{ ...move, next: eventLeaf(p.model, move.nextFeatures), probability: 1 }], s.chosen.depth);
            return { target: trade!.target, exposure: trade!.exposure, quantity: trade!.quantity, gridValueBps: trade!.value * 1e4,
              rewardBps: predicted.reward * 1e4, continuationBps: predicted.continuation * 1e4,
              directValueBps: predicted.value * 1e4, terminalValueBps: predicted.terminal * 1e4,
              interpolationErrorBps: (trade!.value - predicted.value) * 1e4,
              realizedRewardBps: realized.reward * 1e4, realizedBackupBps: realized.value * 1e4 };
          });
        const cash = chosen.find(a => a.target === 0), actual = chosen[0];
        rows.push({ time: t.time, endTime: t.endTime, leaf: t.leaf, exposureBefore: account.exposure,
          realizedReturnBps: t.realizedReturnBps, expectedReturnBps: t.expectedReturnBps, candidates: chosen,
          advantageOverCashBps: cash ? actual.gridValueBps - cash.gridValueBps : null,
          rewardAdvantageOverCashBps: cash ? actual.rewardBps - cash.rewardBps : null,
          continuationAdvantageOverCashBps: cash ? actual.continuationBps - cash.continuationBps : null,
          terminalAdvantageOverCashBps: cash ? actual.terminalValueBps - cash.terminalValueBps : null });
        if (head?.gate) {
          for (const g of groups) g.samples++;
          for (const a of kernel) {
            const g = groups[eventSizeSignGroup(a.return, head.thresholdLogBps)];
            g.predictedMass += a.probability;
            g.predictedLogReturnContribution += a.probability * Math.log1p(a.return) * 1e4;
          }
          const g = groups[eventSizeSignGroup(move.return, head.thresholdLogBps)];
          g.realizedMass++;
          g.realizedLogReturnContribution += Math.log1p(move.return) * 1e4;
        }
      }
    }
    fs.writeFileSync(path.join(output, `${id}-${phase}.json`), JSON.stringify(rows));
    const invested = rows.filter(r => Math.abs(r.candidates[0].exposure) > 0.01);
    phases.push({ phase, samples: rows.length, investedSamples: invested.length, maximumReproductionValueError,
      expectedRewardBps: mean(invested.map(r => r.candidates[0].rewardBps)), realizedRewardBps: mean(invested.map(r => r.candidates[0].realizedRewardBps)),
      expectedValueBps: mean(invested.map(r => r.candidates[0].directValueBps)), realizedBackupBps: mean(invested.map(r => r.candidates[0].realizedBackupBps)),
      meanContinuationAdvantageOverCashBps: mean(invested.map(r => r.continuationAdvantageOverCashBps).filter(v => v !== null)),
      continuationOverridesTerminalCash: invested.filter(r => r.terminalAdvantageOverCashBps !== null && r.terminalAdvantageOverCashBps < 0 && r.advantageOverCashBps > 0).length,
      maximumAbsoluteInterpolationErrorBps: Math.max(0, ...rows.flatMap(r => r.candidates.map((a: any) => Math.abs(a.interpolationErrorBps)))),
      groups: groups.map(g => ({ group: g.group, predictedMass: g.predictedMass / g.samples, realizedMass: g.realizedMass / g.samples,
        predictedConditionalLogReturnBps: g.predictedLogReturnContribution / g.predictedMass,
        realizedConditionalLogReturnBps: g.realizedLogReturnContribution / g.realizedMass,
        predictedContributionBps: g.predictedLogReturnContribution / g.samples, realizedContributionBps: g.realizedLogReturnContribution / g.samples })) });
  }
  fingerprint.update(fs.readFileSync(path.join(source, `${id}-trades.json`)));
  const result = { window: s.window, choice: s.chosen, phases, refitComparison, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result);
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(result));
}
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, sourceHash: fingerprint.digest("hex"),
  compareRefits: process.argv.includes("--compare-refits"),
  caveat: "Completed-outcome diagnostic, not a policy selection or a backtest. Direct values use exact current-account holding transitions and the saved grid continuation. Aggregate event borrowing approximates the minute execution path. Realized backups retain forecast continuation and are not realized full-horizon returns." }, null, 2));
