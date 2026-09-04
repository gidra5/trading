/** Inspect the best active new-law depth chosen on prior origins, even when
 * the cash gate rejects it. No result here changes policy selection. */
import fs from "node:fs";
import path from "node:path";
import { decideEvent, eventActionValues, eventHolding, restoreEventPolicy,
  type EventAccount, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventLeaf, observeMove, type MoveAtom } from "../packages/bot-algo/src/event-distribution.js";
import { eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
if (!arg("source") || !arg("output") || fs.existsSync(output)) throw new Error("Specify saved replay and new audit output");
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const inspectActions = process.argv.includes("--actions");
const config = read(source, "config.json"), selection = read(source, "selection.json"), original = read(config.source, "config.json");
if (config.contract !== "event-volatility-law-policy-v1") throw new Error("Requires volatility-law replay");
const choice = selection.ranking.find((r: any) => r.choice === "joint-volatility" && r.trades > 0);
if (!choice) throw new Error("No active new-law candidate");
const window = config.window, phases = [...eventRefitOrigins(window.startTime, original.foldCount, original.foldDays), { ...window, id: "final" }];
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, choice, inspectActions,
  caveat: "Diagnostic only: disable the cash gate at the best active calibration depth; no choice or promotion based on these test results. Event PnL groups omit terminal settlement after the final event. Action attribution uses completed events and close-price aggregate borrowing; realized backups retain the saved forecast continuation, not realized later returns." }, null, 2));
fs.copyFileSync(__filename, path.join(output, "audit-source.ts"));
const c = loadEventCandles(phases[0].startTime - 2 * DAY, window.endTime + DAY), results = [];
const byTime = new Map(c.map((candle, i) => [candle.openTime + 60_000, i]));
const mean = (values: number[]) => values.length ? values.reduce((s, x) => s + x, 0) / values.length : null;
function directValue(p: EventPolicy, account: EventAccount, trade: EventAccount, kernel: readonly MoveAtom[], depth: number) {
  let reward = Math.log(trade.equity / account.equity), continuation = 0, settlement = 0;
  for (const atom of kernel) {
    if (!atom.probability) continue;
    const held = eventHolding(trade.exposure, atom, p.costs);
    if (held.liquidated) return { reward: -Infinity, continuation: -Infinity, terminal: -Infinity, value: -Infinity };
    const terminal = Math.log(1 - Math.abs(held.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4);
    reward += atom.probability * Math.log(held.factor);
    settlement += atom.probability * terminal;
    continuation += atom.probability * (depth > 1 ? decideEvent(p, atom.next, {
      equity: trade.equity * held.factor, price: trade.price * (1 + atom.return), exposure: held.exposure,
    }, depth - 1).value : terminal);
  }
  return { reward, continuation, terminal: reward + settlement, value: reward + continuation };
}
for (const phase of phases) {
  const policy = restoreEventPolicy(read(source, `${phase.id}-policy.json`));
  const replay = replayEventPolicy(c, policy, phase.startTime, phase.endTime, choice.depth, { trace: true });
  if (phase.id !== "final") {
    const old = read(source, `${phase.id}-scores.json`).rows.find((r: any) => r.choice === "joint-volatility" && r.depth === choice.depth);
    if (old.returnPct !== replay.returnPct || old.trades !== replay.trades) throw new Error("Active calibration does not reproduce");
  }
  fs.writeFileSync(path.join(output, `${phase.id}-trades.json`), JSON.stringify(replay.trace));
  const { trace, ...metrics } = replay;
  const actions = inspectActions ? trace.flatMap((raw: any) => {
    if (!raw.orderQuantity) return [];
    const i = byTime.get(raw.time);
    if (i === undefined) throw new Error("Missing decision candle");
    const move = observeMove(c, i, policy.model.clock, policy.model.featureNames)!;
    if (c[move.end].openTime + 60_000 > phase.endTime) return [];
    const account = { equity: raw.equityBefore, price: c[i].close, exposure: raw.exposureBefore };
    const values = eventActionValues(policy, raw.leaf, account, choice.depth);
    const selected = values.find(v => Math.abs(v.quantity - raw.order.quantity) < 1e-12);
    if (!selected || Math.abs(selected.value - raw.order.value) > 1e-12) throw new Error("Action value does not reproduce");
    const hold = values.find(v => v.quantity === 0);
    const candidates = [selected, hold, values.find(v => v.target === 0)]
      .filter((v, j, all) => v && all.indexOf(v) === j).map(trade => {
        const predicted = directValue(policy, account, trade!, policy.model.kernels[raw.leaf], choice.depth);
        const realized = directValue(policy, account, trade!, [{ ...move, probability: 1, next: eventLeaf(policy.model, move.nextFeatures) }], choice.depth);
        return { quantity: trade!.quantity, exposure: trade!.exposure, orderLogCost: trade!.orderLogCost,
          gridValue: trade!.value, predicted, realized };
      });
    const actual = candidates[0], unchanged = candidates.find(v => v.quantity === 0);
    const difference = (a: number, b: number) => (a - b) * 1e4;
    return [{ time: raw.time, endTime: raw.endTime, leaf: raw.leaf, exposureBefore: raw.exposureBefore,
      kind: Math.abs(account.exposure) < 0.01 ? "entry" : Math.abs(selected.exposure) < 0.01 ? "exit"
        : account.exposure * selected.exposure < 0 ? "reversal" : "resize",
      forcedCap: !hold, realizedReturnBps: raw.realizedReturnBps, expectedReturnBps: raw.expectedReturnBps, candidates,
      advantage: unchanged ? {
        gridBps: difference(actual.gridValue, unchanged.gridValue),
        rewardBps: difference(actual.predicted.reward, unchanged.predicted.reward),
        continuationBps: difference(actual.predicted.continuation, unchanged.predicted.continuation),
        terminalBps: difference(actual.predicted.terminal, unchanged.predicted.terminal),
        realizedRewardBps: difference(actual.realized.reward, unchanged.realized.reward),
        realizedBackupBps: difference(actual.realized.value, unchanged.realized.value),
      } : null }];
  }) : [];
  if (inspectActions) fs.writeFileSync(path.join(output, `${phase.id}-actions.json`), JSON.stringify(actions));
  const summarizeActions = (rows: typeof actions) => ({ orders: rows.length, feasibleHold: rows.filter(r => r.advantage).length,
    orderCostBps: -rows.reduce((s, r) => s + r.candidates[0].orderLogCost * 1e4, 0),
    meanAdvantage: Object.fromEntries(["gridBps", "rewardBps", "continuationBps", "terminalBps", "realizedRewardBps", "realizedBackupBps"]
      .map(key => [key, mean(rows.filter(r => r.advantage).map(r => (r.advantage as any)[key]))])),
    continuationOverridesTerminalHold: rows.filter(r => r.advantage && r.advantage.terminalBps < 0).length,
    positiveRealizedRewardAdvantage: rows.filter(r => r.advantage && r.advantage.realizedRewardBps > 0).length });
  const actionSummary = inspectActions ? { all: summarizeActions(actions),
    byKind: Object.fromEntries(["entry", "exit", "reversal", "resize"].map(kind => [kind, summarizeActions(actions.filter(r => r.kind === kind))])),
    byAdvantage: [0, 0.1, 1, 5, 20].map((lower, i, edges) => ({ lower, upper: edges[i + 1] ?? null,
      ...summarizeActions(actions.filter(r => r.advantage && r.advantage.gridBps >= lower && r.advantage.gridBps < (edges[i + 1] ?? Infinity))) })),
    maximumInterpolationErrorBps: Math.max(0, ...actions.flatMap(r => r.candidates.map(v => Math.abs(v.gridValue - v.predicted.value) * 1e4))),
  } : undefined;
  const groups = [0, 1].map(band => {
    const rows = trace.filter((r: any) => r.leaf % 2 === band) as any[];
    return { band, events: rows.length, orders: rows.filter(r => r.orderQuantity).length,
      eventPnl: rows.reduce((s, r) => s + r.equityAfter - r.equityBefore, 0),
      forecastReturnBps: rows.reduce((s, r) => s + r.expectedReturnBps, 0) / rows.length,
      realizedReturnBps: rows.reduce((s, r) => s + r.realizedReturnBps, 0) / rows.length };
  });
  const result = { phase, depth: choice.depth, metrics, groups, actionSummary };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ phase: phase.id, returnPct: metrics.returnPct, drawdown: metrics.maxDrawdownPct, trades: metrics.trades, fees: metrics.fees, longPnl: metrics.longPnl, shortPnl: metrics.shortPnl, groups,
    actions: actionSummary?.all, maximumInterpolationErrorBps: actionSummary?.maximumInterpolationErrorBps }));
}
