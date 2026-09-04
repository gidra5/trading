/** One bounded value-aware improvement of an existing probability head. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { buildEventOutcomeLookahead, decideEvent, eventHolding, eventOutcomeHoldingValues, restoreEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, predictEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { eventValueHeadMse, trainEventValueHead, type EventValueHeadRow } from "../packages/bot-algo/src/event-value-head.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), plannerId = arg("planner-source"), outputId = arg("output");
if (!forecastId || !plannerId || !outputId) throw new Error("Specify frozen forecast/planner sources and a new output");
const forecast = path.resolve(root, "data/benchmarks", forecastId), planner = path.resolve(root, "data/benchmarks", plannerId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), pc = read(planner, "config.json"), source: string = fc.source;
const sc = read(source, "config.json"), summary = read(source, "summary.json");
if (fc.contract !== "event-size-sign-forecast-screen-v1" || pc.forecast !== forecast || !pc.projectContinuation
  || sc.sampling !== "chain" || sc.invertAugment || sc.calibrateMean || sc.onlineScale || sc.honestyFraction) throw new Error("Incompatible fixed model sources");
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
const saved = summary.map((s: any) => {
  const bytes = fs.readFileSync(path.join(source, `${s.window.id}-model.json`)); hash.update(bytes); return JSON.parse(bytes.toString());
});
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Base models changed after forecast selection");
const ids = arg("windows").split(",").filter(Boolean);
if (ids.some(id => !summary.some((s: any) => s.window.id === id))) throw new Error("Unknown window");
const penalties = [0.01, 0.1, 1], depths = [1, 4, 8];
fs.mkdirSync(output, { recursive: true });
const parentHash = createHash("sha256");
for (const dir of [forecast, planner]) {
  parentHash.update(fs.readFileSync(path.join(dir, "config.json")));
  for (const s of summary) parentHash.update(fs.readFileSync(path.join(dir, `${s.window.id}-${dir === forecast ? "model" : "continuation"}.json`)));
}
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ ...fc, frozenHeadSource: forecast, plannerSource: planner,
  parentHash: parentHash.digest("hex"), windows: ids.length ? ids : summary.map((s: any) => s.window.id),
  model: "One value-aware correction to the fixed size/sign logistic components; delta parameters including intercepts penalized around the incumbent",
  valueProbes: { depths, exposures: "both signed maximum exposures", center: "subtract cash continuation", normalization: "fit-only RMS target advantage" }, penalties,
  selection: "Preceding calibration squared Bellman-advantage error; unchanged incumbent wins ties; subsequent policy comparison still required",
  caveat: "One fixed-planner value-aware update, not IterVAML convergence. Models use the original fit population; calibration selects the correction; test targets are diagnostic only." }, null, 2));
const files = ["scripts/screen-event-value-head.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-log-policy.ts",
  "packages/bot-algo/src/event-value-head.ts", "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (let index = 0; index < summary.length; index++) {
  const s = saved[index], { window } = summary[index], id = window.id;
  if (ids.length && !ids.includes(id)) continue;
  const started = performance.now(), h = read(forecast, `${id}-model.json`), projected = read(planner, `${id}-continuation.json`);
  if (!h.selected || h.selected.gate || !h.head?.gate || h.head.valueFit || h.selectionTrainingEnd !== s.selectionTrainingEnd
    || h.trainEnd !== s.trainEnd || s.selectionTrainingEnd >= s.policyCalibrationStart || s.trainEnd > window.startTime
    || s.calibrationEnd > window.startTime || id.startsWith("fit-")) throw new Error("Invalid value-head fit boundaries or incumbent");
  const trainStart = window.startTime - (sc.calibrationDays + sc.trainDays) * DAY;
  const c = loadEventCandles(trainStart - 2 * DAY, window.endTime + DAY);
  const excluded = sc.trainingIsolation === "causal" ? [window] : sc.excludedWindows;
  const samples = (start: number, end: number, exclude: typeof excluded) => makeSamples(c, sc.clock, start, end, exclude, sc.stride, "chain", sc.featureNames);
  const prepare = (p: EventPolicy, head: EventSizeSignHead) => {
    if (p.tables.length < Math.max(...depths)) throw new Error("Missing probe depths");
    const lookahead = buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, head.thresholdLogBps));
    return (moves: MoveSample[]) => {
      let memory: EventCompletedHistory | undefined, previousEnd = -1;
      const rows: EventValueHeadRow[] = [], origins: number[] = [];
      for (const move of moves) {
        const at = c[move.start].openTime + 60_000;
        if (h.selected.eventHistory && move.start !== previousEnd) memory = new EventCompletedHistory(at);
        const features = [...move.features, ...(h.selected.fastVolatility ? eventFastVolatilityFeatures(c, move.start) : []), ...(memory?.features(at) ?? [])];
        memory?.observe({ ...move, originTime: at, availableAt: c[move.end].openTime + 60_000 }, c[move.end].openTime + 60_000);
        previousEnd = move.end;
        const leaf = eventLeaf(p.model, move.features), next = eventLeaf(p.model, move.nextFeatures), mass = lookahead.masses[leaf];
        // Missing conditional paths trigger the unchanged-law fallback at inference.
        if (mass.slice(0, 4).some(v => !v)) continue;
        const equity = p.equities[2], price = c[move.start].close, nextPrice = c[move.end].close;
        const active = mass.slice(0, 4).reduce((s, v) => s + v, 0);
        const coefficients: number[][] = [], offsets: number[] = [], targets: number[] = [];
        for (const depth of depths) {
          const cashValues = eventOutcomeHoldingValues(lookahead, leaf, { equity, price, exposure: 0 }, depth);
          const nextCash = depth > 1 ? decideEvent(p, next, { equity, price: nextPrice, exposure: 0 }, depth - 1).value : 0;
          for (const exposure of [-p.costs.maxLeverage, p.costs.maxLeverage]) {
            const values = eventOutcomeHoldingValues(lookahead, leaf, { equity, price, exposure }, depth);
            const hold = eventHolding(exposure, move, p.costs);
            const terminal = Math.log(1 - Math.abs(hold.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 1e4);
            const continuation = depth > 1 ? decideEvent(p, next, { equity: equity * hold.factor, price: nextPrice, exposure: hold.exposure }, depth - 1).value : terminal;
            coefficients.push(values.slice(0, 4).map((v, g) => active * (v - cashValues[g])));
            offsets.push(mass[4] ? mass[4] * (values[4] - cashValues[4]) : 0);
            targets.push(hold.liquidated ? -Infinity : Math.log(hold.factor) + continuation - nextCash);
          }
        }
        if ([...coefficients.flat(), ...offsets, ...targets].some(v => !Number.isFinite(v))) throw new Error("Value probe encounters ruin; choose feasible exposure probes before fitting");
        rows.push({ features, coefficients, offsets, targets }); origins.push(at);
      }
      return { rows, origins };
    };
  };
  const selectionRows = prepare(restoreEventPolicy(projected.selectionPolicy), h.selectionHead);
  const fit = samples(trainStart, s.selectionTrainingEnd, excluded), calibration = samples(s.policyCalibrationStart, s.calibrationEnd, excluded);
  if (fit.length !== summary[index].selectionFitSamples) throw new Error("Changed original fit population");
  const training = selectionRows(fit), validation = selectionRows(calibration);
  const candidates = [{ penalty: null as number | null, head: h.selectionHead as EventSizeSignHead },
    ...penalties.map(penalty => ({ penalty, head: trainEventValueHead(h.selectionHead, training.rows, penalty) }))];
  const scored = candidates.map(candidate => ({ ...candidate, mse: eventValueHeadMse(candidate.head, validation.rows) })).sort((a, b) => a.mse - b.mse);
  const chosen = scored[0], selected = { ...h.selected, valuePenalty: chosen.penalty };
  fs.writeFileSync(path.join(output, `${id}-selection.json`), JSON.stringify({ window, selected, scores: scored.map(({ head: _, ...r }) => r) }, null, 2));
  const finalRows = prepare(restoreEventPolicy(projected.policy), h.head), finalFit = samples(s.trainStart, s.trainEnd, excluded);
  if (finalFit.length !== summary[index].fitSamples) throw new Error("Changed final fit population");
  const head = chosen.penalty === null ? h.head : trainEventValueHead(h.head, finalRows(finalFit).rows, chosen.penalty);
  fs.writeFileSync(path.join(output, `${id}-model.json`), JSON.stringify({ selected, head, selectionHead: chosen.head,
    trainEnd: s.trainEnd, selectionTrainingEnd: s.selectionTrainingEnd, finalTargetEnd: c[finalFit.at(-1)!.end].openTime + 60_000 }, null, 2));
  const test = finalRows(samples(window.startTime, window.endTime, []));
  const testMse = eventValueHeadMse(head, test.rows), baseTestMse = eventValueHeadMse(h.head, test.rows);
  fs.writeFileSync(path.join(output, `${id}-predictions.json`), JSON.stringify(test.rows.map((r, i) => ({ time: test.origins[i], probabilities: predictEventSizeSigns(head, r.features) }))));
  const result = { window, selected, calibration: scored.map(({ head: _, ...r }) => r), test: { samples: test.rows.length, mse: testMse, incumbentMse: baseTestMse,
    skill: 1 - testMse / baseTestMse }, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(result));
}
