/** Rolling-origin comparison of event-policy refit rules. Models at each
 * origin only consume completed earlier targets; the final inspector window
 * cannot affect candidate selection. This is a research suite, not a sealed test. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { trainEventRunDistribution } from "../packages/bot-algo/src/event-run-model.js";
import { buildEventOutcomeLookahead, buildEventPolicy, serializeEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, predictEventSizeSigns, projectEventSizeSigns,
  trainEventSizeGate, trainEventSizeSign, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";

const DAY = 86_400_000, root = path.resolve(__dirname, "..");
interface Window { id: string; startTime: number; endTime: number; }
interface OriginScore { logGrowth: number; maxDrawdownPct: number; }
export function eventRefitOrigins(testStart: number, folds: number, foldDays: number): Window[] {
  if (!Number.isSafeInteger(testStart) || !Number.isInteger(folds) || folds < 2 || folds > 20
    || !Number.isInteger(foldDays) || foldDays < 1 || foldDays > 90) throw new Error("Invalid rolling-origin boundaries");
  return Array.from({ length: folds }, (_, i) => {
    const startTime = testStart - (folds - i) * foldDays * DAY;
    if (startTime < DAY) throw new Error("Insufficient origin history");
    return { id: `origin-${new Date(startTime).toISOString().slice(0, 10)}`, startTime, endTime: startTime + foldDays * DAY };
  });
}
export function eventOriginScore(rows: readonly OriginScore[], riskPenalty: number) {
  if (!rows.length || !(riskPenalty >= 0) || !Number.isFinite(riskPenalty)
    || rows.some(r => !Number.isFinite(r.logGrowth) || !Number.isFinite(r.maxDrawdownPct) || r.maxDrawdownPct < 0)) throw new Error("Invalid origin scores");
  const scores = rows.map(r => r.logGrowth - riskPenalty * r.maxDrawdownPct / 100);
  const score = scores.reduce((s, v) => s + v, 0) / scores.length;
  return { score, scores, positiveOrigins: rows.filter(r => r.logGrowth > 0).length,
    minimumScore: Math.min(...scores), maximumScore: Math.max(...scores),
    scoreStd: Math.sqrt(scores.reduce((s, v) => s + (v - score) ** 2, 0) / scores.length),
    meanLogGrowth: rows.reduce((s, r) => s + r.logGrowth, 0) / rows.length,
    worstDrawdownPct: Math.max(...rows.map(r => r.maxDrawdownPct)) };
}

type Choice = "base" | "head-base" | "head-projected" | "lag-base" | "lag-projected";
const choices: Choice[] = ["base", "head-base", "head-projected", "lag-base", "lag-projected"];

export async function main() {
  const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
  const forecastId = arg("forecast"), outputId = arg("output"), foldCount = Number(arg("folds", "3")), foldDays = Number(arg("fold-days", "21"));
  if (!forecastId || !outputId) throw new Error("Specify frozen forecast settings and a new output");
  const forecast = path.resolve(root, "data/benchmarks", forecastId), output = path.resolve(root, "data/benchmarks", outputId);
  const read = (dir: string, name: string) => JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
  if (fs.existsSync(output)) throw new Error("Choose a new output directory");
  const fc = read(forecast, "config.json"), source: string = fc.source, sc = read(source, "config.json");
  if (fc.contract !== "event-size-sign-forecast-screen-v1" || sc.trainingIsolation !== "causal" || sc.sampling !== "chain"
    || sc.runDirectionPrior === undefined || sc.invertAugment || sc.calibrateMean || sc.onlineScale || sc.honestyFraction)
    throw new Error("Rolling refit comparison requires causal run-conditioned models and fixed size/sign heads");
  const summaries = read(source, "summary.json"), ids = arg("windows").split(",").filter(Boolean);
  if (ids.some(id => !summaries.some((s: any) => s.window.id === id))) throw new Error("Unknown window");
  const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
  for (const s of summaries) hash.update(fs.readFileSync(path.join(source, `${s.window.id}-model.json`)));
  if (hash.digest("hex") !== fc.sourceHash) throw new Error("Source models changed since forecast selection");
  const windows = summaries.filter((s: any) => !ids.length || ids.includes(s.window.id));
  for (const s of windows) eventRefitOrigins(s.window.startTime, foldCount, foldDays);
  const headLagDays = sc.calibrationDays + 1;
  fs.mkdirSync(output, { recursive: true });
  const settings = windows.map((s: any) => {
    const h = read(forecast, `${s.window.id}-model.json`);
    if (!h.selected || !h.head?.gate || h.selected.gate || h.head.valueFit || s.window.id.startsWith("fit-")) throw new Error("Incompatible incumbent settings");
    return { window: s.window, treeDepth: s.treeDepth, head: h.selected, componentPenalties: [h.head.gate.penalty, h.head.ordinarySign.penalty, h.head.largeSign.penalty] };
  });
  fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "rolling-event-refit-v1", source, forecast, sourceHash: fc.sourceHash,
    foldCount, foldDays, headLagDays, settings, choices, depths: sc.maxDepth, costs: sc.costs, clock: sc.clock,
    selection: "Mean per-origin log growth minus the configured drawdown penalty; all completed pre-window origins equally weighted; cash eligible; original policy wins ties",
    training: "Fresh model and head use trainDays ending strictly before each origin. Delayed head uses trainDays - 1 ending headLagDays before each origin. Same rules at validation and final deployment.",
    scope: "Compare refit rules within incumbent tree/head hyperparameters; their earlier research selection is not an untouched nested outer test. Each historical fit uses only past data.",
    caveat: "Inspector windows have been repeatedly inspected. Origin results select candidates; final scored outcomes never enter this selection. Separate origins start in cash and include terminal settlement." }, null, 2));
  const files = ["scripts/research-event-refits.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-distribution.ts",
    "packages/bot-algo/src/event-run-model.ts", "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts",
    "packages/bot-algo/src/event-completed-history.ts", "packages/bot-algo/src/event-log-policy.ts"];
  fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
  const results = [];
  for (const setting of settings) {
    const started = performance.now(), window: Window = setting.window;
    const origins = eventRefitOrigins(window.startTime, foldCount, foldDays);
    const c = loadEventCandles(origins[0].startTime - (sc.trainDays + headLagDays + 2) * DAY, window.endTime + DAY);
    const modelStartPrice = (origin: number) => {
      const i = c.findIndex(v => v.openTime + 60_000 === origin);
      if (i < 1440) throw new Error("Missing completed origin candle");
      return c[i].close;
    };
    const samples = (from: number, to: number) => {
      const rows = makeSamples(c, sc.clock, from, to, [window], sc.stride, "chain", sc.featureNames);
      if (rows.length < Math.max(200, sc.minLeaf * 2) || rows.some(r => c[r.end].openTime + 60_000 >= to)) throw new Error("Insufficient or future-contaminated refit population");
      return rows;
    };
    const augment = (rows: MoveSample[]) => {
      let previousEnd = -1, memory: EventCompletedHistory | undefined;
      return rows.map(row => {
        const at = c[row.start].openTime + 60_000;
        if (setting.head.eventHistory && row.start !== previousEnd) memory = new EventCompletedHistory(at);
        const features = [...row.features, ...(setting.head.fastVolatility ? eventFastVolatilityFeatures(c, row.start) : []), ...(memory?.features(at) ?? [])];
        memory?.observe({ ...row, originTime: at, availableAt: c[row.end].openTime + 60_000 }, c[row.end].openTime + 60_000);
        previousEnd = row.end;
        return { ...row, features };
      });
    };
    const trainHead = (rows: MoveSample[]) => {
      const [gatePenalty, ordinaryPenalty, largePenalty] = setting.componentPenalties;
      if (ordinaryPenalty !== largePenalty) throw new Error("Unsupported differing sign-component penalties");
      if (setting.head.historyGateOnly) {
        const width = sc.featureNames.length + (setting.head.fastVolatility ? 3 : 0);
        const base = trainEventSizeSign(rows.map(r => ({ ...r, features: r.features.slice(0, width) })), ordinaryPenalty, setting.head.quantile);
        return trainEventSizeGate(rows, base, gatePenalty);
      }
      if (gatePenalty !== ordinaryPenalty) throw new Error("Unsupported differing component penalties");
      return trainEventSizeSign(rows, gatePenalty, setting.head.quantile);
    };
    const fit = (origin: number, id: string) => {
      const start = performance.now();
      const trainStart = origin - sc.trainDays * DAY, trainEnd = origin;
      const lagTrainEnd = origin - headLagDays * DAY, lagTrainStart = lagTrainEnd - (sc.trainDays - 1) * DAY;
      const rows = samples(trainStart, trainEnd), inputs = augment(rows), lagRows = samples(lagTrainStart, lagTrainEnd);
      const head = trainHead(inputs), lagHead = trainHead(augment(lagRows));
      const model = trainEventRunDistribution(rows, sc.clock, { maxDepth: setting.treeDepth, minLeaf: sc.minLeaf, prior: sc.prior,
        criterion: sc.criterion, directionPrior: sc.runDirectionPrior });
      const policyOptions = { depths: sc.maxDepth, referenceEquity: 10_000, referencePrice: modelStartPrice(origin), actionSteps: sc.actionSteps };
      const base = buildEventPolicy(model, sc.costs, policyOptions);
      const projected = buildEventPolicy(projectEventSizeSigns(model, head.thresholdLogBps,
        rows.map((r, i) => ({ features: r.features, probabilities: predictEventSizeSigns(head, inputs[i].features) })), setting.head.blend), sc.costs, policyOptions);
      const headOptions = (p: EventPolicy, h: EventSizeSignHead) => ({ sizeSign: { head: h, blend: setting.head.blend,
        fastVolatility: setting.head.fastVolatility, eventHistory: setting.head.eventHistory,
        lookahead: buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, h.thresholdLogBps)) } });
      const policies = { base, "head-base": base, "head-projected": projected, "lag-base": base, "lag-projected": projected };
      const options = { base: {}, "head-base": headOptions(base, head), "head-projected": headOptions(projected, head),
        "lag-base": headOptions(base, lagHead), "lag-projected": headOptions(projected, lagHead) };
      const support = (rows: MoveSample[]) => ({ samples: rows.length, firstInput: c[rows[0].start - 1440].openTime,
        firstOrigin: c[rows[0].start].openTime + 60_000, lastTarget: c[rows.at(-1)!.end].openTime + 60_000 });
      const metadata = { origin, trainStart, trainEnd, lagTrainStart, lagTrainEnd, support: support(rows), lagSupport: support(lagRows) };
      fs.writeFileSync(path.join(output, `${window.id}-${id}-model.json`), JSON.stringify({ ...metadata, head, lagHead,
        base: serializeEventPolicy(base), projected: serializeEventPolicy(projected) }));
      console.log(JSON.stringify({ event: "origin-fit", window: window.id, id, ...metadata, elapsedSec: (performance.now() - start) / 1000 }));
      return { policies, options, metadata };
    };
    const folds = origins.map(origin => {
      const fitted = fit(origin.startTime, origin.id), start = performance.now();
      const rows = choices.flatMap(choice => Array.from({ length: sc.maxDepth }, (_, d) => {
        const depth = d + 1, { trace: _trace, ...metrics } = replayEventPolicy(c, fitted.policies[choice], origin.startTime, origin.endTime, depth, fitted.options[choice]);
        return { choice, depth, ...metrics };
      }));
      fs.writeFileSync(path.join(output, `${window.id}-${origin.id}-scores.json`), JSON.stringify({ origin, rows }, null, 2));
      console.log(JSON.stringify({ event: "origin-scored", window: window.id, origin, elapsedSec: (performance.now() - start) / 1000 }));
      return { origin, fitted, rows };
    });
    const ranking = choices.flatMap(choice => Array.from({ length: sc.maxDepth }, (_, d) => {
      const depth = d + 1, rows = folds.map(f => f.rows.find(r => r.choice === choice && r.depth === depth)!);
      return { choice, depth, ...eventOriginScore(rows, sc.riskPenalty), trades: rows.reduce((s, r) => s + r.trades, 0) };
    })).sort((a, b) => b.score - a.score);
    const chosen = ranking[0], cash = chosen.score <= 0;
    fs.writeFileSync(path.join(output, `${window.id}-selection.json`), JSON.stringify({ window, chosen, cash, origins, ranking }, null, 2));
    // No final fit or scored-window evaluation occurs before this selection.
    const final = fit(window.startTime, "final");
    const replay = replayEventPolicy(c, final.policies[chosen.choice], window.startTime, window.endTime, chosen.depth,
      { ...final.options[chosen.choice], cash, trace: true });
    fs.writeFileSync(path.join(output, `${window.id}-trades.json`), JSON.stringify(replay.trace));
    for (const fold of folds) {
      const replay = replayEventPolicy(c, fold.fitted.policies[chosen.choice], fold.origin.startTime, fold.origin.endTime, chosen.depth,
        { ...fold.fitted.options[chosen.choice], cash, trace: true });
      fs.writeFileSync(path.join(output, `${window.id}-${fold.origin.id}-chosen-trades.json`), JSON.stringify(replay.trace));
    }
    const { trace: _, ...test } = replay;
    const result = { window, chosen, cash, treeDepth: setting.treeDepth, test, elapsedSec: (performance.now() - started) / 1000 };
    results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
    console.log(JSON.stringify({ event: "rolling-refit-result", ...result, test: { ...test, daily: undefined, adaptationPairs: undefined } }));
  }
}

if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
