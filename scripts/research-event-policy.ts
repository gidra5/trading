import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { readCandleShardReferenceSync } from "../packages/storage/src/candles.js";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import { calibrateEventMean, distributionMetrics, EVENT_FEATURES, EVENT_SECOND_FEATURES, EVENT_PATH_FEATURES, EVENT_RUN_FEATURES, eventFeatures, eventLeaf, eventHiddenNext, observeMove, scaleEventMean, trainEventDistribution, trainEventForest, trainEventProjection, trainEventBoost,
  eventCandleIntervalMs, eventFeatureWarmup, eventBaseFeatures,
  type EventCandle, type EventClock, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { buildEventPolicy, chooseEventTrade, decideEvent, decideEventKernel, decideEventOutcomes, decideEventSign, DEFAULT_EVENT_COSTS, eventCapAllowsTrade, eventHolding, serializeEventPolicy,
  type EventPolicy, type EventOutcomeLookahead, type EventKernelLookahead } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSignMass, predictEventSign, reweightEventSigns, type EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { eventFastVolatilityFeatures, mixEventSizeSigns, predictEventSizeSigns, reweightEventSizeSigns, selectEventSizeSignComponents,
  type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { eventSeverityMeans, predictEventSeverity, tiltEventSeverity, type EventSeverityHead } from "../packages/bot-algo/src/event-severity.js";
import { EventSecondBasis } from "./event-second-basis.js";
import { trainEventHidden } from "../packages/bot-algo/src/event-hidden.js";
import { trainEventRunDistribution } from "../packages/bot-algo/src/event-run-model.js";
import { decideFittedEvent, decideFittedEventControllers, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { decideEventOneStep, type EventTerminal } from "../packages/bot-algo/src/event-one-step.js";
import { prepareEventTwoStep } from "../packages/bot-algo/src/event-two-step.js";
import { prepareEventThreeStep } from "../packages/bot-algo/src/event-three-step.js";
import { prepareEventExecutionOneStep, type EventExecutionAtom } from "../packages/bot-algo/src/event-execution-one-step.js";
import { EventPositionLedger } from "../packages/bot-algo/src/event-positions.js";
import { mergeEventSourceRanges, type EventSourceRange } from "./event-fit-periods.js";
import { eventMarketUnavailable, eventAvailabilitySourceStart, markUnavailableEventSeconds, validateEventMarketClosures, validateEventSourceSecond,
  type EventMarketClosure, type EventSourceSecond } from "../packages/bot-algo/src/event-market-availability.js";

const DAY = 86_400_000;
const root = path.resolve(__dirname, "..");
export function invertEventCandles(c: readonly EventCandle[]): EventCandle[] {
  // Rebuild the actual counterfactual chart. Merely negating event returns would
  // change first-passage times and silently corrupt event labels.
  return c.map(v => ({ ...v, open: 1 / v.open, high: 1 / v.low, low: 1 / v.high, close: 1 / v.close }));
}
interface Range { id: string; startTime: number; endTime: number; }
export function overlaps(start: number, end: number, ranges: readonly Range[]): boolean {
  return ranges.some(r => start < r.endTime && end >= r.startTime);
}
export function makeSamples(c: readonly EventCandle[], clock: EventClock, start: number, end: number,
  excluded: readonly Range[], stride: number, sampling: "stride" | "chain" = "stride",
  featureNames: readonly string[] = EVENT_FEATURES): MoveSample[] {
  if (c.some(row => row.carriedMark)) throw new Error("Replay-only carried marks cannot be used as observed training samples");
  const rows: MoveSample[] = [];
  const interval = eventCandleIntervalMs(clock), warmup = eventFeatureWarmup(clock, featureNames);
  for (let i = warmup; i < c.length - 1;) {
    const time = c[i].openTime + interval;
    if (time < start || time >= end || c[i].openTime - c[i - warmup].openTime !== warmup * interval) { i++; continue; }
    const sample = observeMove(c, i, clock, featureNames);
    // A censored final chain has no observed successor at which to restart.
    // Advancing by stride here would invent a new origin inside that event.
    const remaining = c.length - 1 - i;
    if (!sample && sampling === "chain" && remaining < clock.maxCandles
      && c[c.length - 1].openTime - c[i].openTime === remaining * interval
      && !observeMove(c, i, clock, eventBaseFeatures(clock))) break;
    if (!sample || c[sample.end].openTime + interval >= end) { i += sampling === "chain" && sample ? sample.end - i : stride; continue; }
    // Purge the ENTIRE input and target support, not just the decision timestamp.
    if (!overlaps(c[i - warmup].openTime, c[sample.end].openTime + interval, excluded)) rows.push(sample);
    i += sampling === "chain" ? sample.end - i : stride;
  }
  return rows;
}

export interface EventPolicyUpdate { at: number; policy: EventPolicy; }
/** Apply the replay's terminal market clips, retaining sub-minimum dust. */
export function settleEventReplayAccount(equity: number, quantity: number, price: number, costs: EventPolicy["costs"]) {
  const notional = Math.abs(quantity) * price;
  if (quantity && equity > 0 && notional >= costs.minNotional - 1e-8 && Math.abs(quantity) >= costs.minQuantity - 1e-12) {
    const fee = notional * (costs.feeBps + costs.slippageBps) / 10000;
    return { equity: equity - fee, quantity: 0, fee, orders: Math.ceil(notional / costs.maxNotional), dust: 0 };
  }
  return { equity, quantity, fee: 0, orders: 0, dust: quantity && equity > 0 ? notional : 0 };
}
export function replayEventPolicy(c: readonly EventCandle[], p: EventPolicy, start: number, end: number,
  depth: number, options: { equity?: number; initialQuantity?: number; trace?: boolean; cash?: boolean;
    /** Diagnostic lifecycle attribution; defaults to trace, never changes decisions. */
    positionAttribution?: boolean;
    /** Explicit execution scenario. Only completed status reaches the controller. */
    marketClosures?: readonly EventMarketClosure[];
    onDecision?: (row: Record<string, unknown>) => void;
    /** Flat-entry execution experiment: freeze quote turnover at the decision close. */
    quoteEntries?: true;
    /** Exact one-event lot search under the frozen joint kernel; explicit terminal contract. */
    oneStepTerminal?: EventTerminal;
    /** Full empirical execution paths with an unchanged original joint projection. */
    executionOneStep?: readonly (readonly (EventExecutionAtom & { next: number })[])[];
    /** Receding two-event planning; each trace retains its finite-horizon value gap. */
    twoStep?: { terminal: "marked" | "friction"; maxEvaluations: number; tolerance?: number;
      /** Reuse the recursive continuous relaxation to tighten root bounds. */
      globalUpper?: true;
      /** Count down H2 -> H1 on observed states before starting the next block. */
      countdown?: true; initialDepth?: 1 | 2 };
    /** Receding H3 policy; a budget stop retains the unresolved global gap. */
    threeStep?: { terminal: "marked" | "friction"; maxEvaluations: number; tolerance?: number;
      maxRootEvaluations?: number; shadowPoints?: number };
    updates?: readonly EventPolicyUpdate[];
    sign?: { head: EventSignHead; blend: number; lookahead: EventOutcomeLookahead;
      observations?: ReadonlyMap<number, { availableAt: number; values: number[] }> };
    sizeSign?: { head: EventSizeSignHead; blend: number; fastVolatility?: boolean; eventHistory?: boolean; components?: readonly number[]; lookahead: EventOutcomeLookahead;
      observations?: ReadonlyMap<number, { availableAt: number; values: number[] }> };
    severity?: { head: EventSeverityHead; blend: number; lookahead: EventKernelLookahead };
    fitted?: { policy: FittedEventValue; observations: ReadonlyMap<number, { availableAt: number; values: number[] }>;
      /** Advance a minimum-turnover option, or sampled H2 followed by H1, before replanning.
       * Minimum-turnover cash can replan immediately; sampled cash commits to its next H1. */
      replanEvery?: number;
      /** Timing ablation: reconsider H2 at every flat state, even after sampled waiting. */
      replanCash?: true;
      /** Controller memory before the first decision when resuming an observed account. */
      initialState?: { remaining: number; controller: 0 | 1 };
      /** Alternative H2 controller: act with its shared H1 at the next event, including after waiting in cash. */
      sampledAlternative?: FittedEventValue };
    adaptive?: { policies: EventPolicy[]; scales: number[]; window: number; initialPairs?: Array<[number, number]> } } = {}) {
  const interval = eventCandleIntervalMs(p.model.clock), barMinutes = interval / 60000;
  const warmup = eventFeatureWarmup(p.model.clock, p.model.featureNames);
  if (options.executionOneStep && (interval !== 1000 || !options.oneStepTerminal
    || options.oneStepTerminal === "friction" || options.quoteEntries
    || options.executionOneStep.length !== p.model.kernels.length)) throw new Error("Execution-law replay requires native exact H1 and base requests");
  if (options.executionOneStep?.some((kernel, leaf) => kernel.length !== p.model.kernels[leaf].length || kernel.some((atom, j) => {
    const old = p.model.kernels[leaf][j], path = atom.path;
    return atom.probability !== old.probability || atom.next !== old.next || path.closeRatio - 1 !== old.return
      || path.seconds / 60 !== old.duration || Math.min(1, path.lowRatio) - 1 !== old.low || Math.max(1, path.highRatio) - 1 !== old.high
      || (Object.keys(p.costs) as Array<keyof typeof p.costs>).some(key => path.costs[key] !== p.costs[key]);
  }))) throw new Error("Execution law changed the frozen joint forecast or account costs");
  const execution = options.executionOneStep?.map(kernel => prepareEventExecutionOneStep(kernel, options.oneStepTerminal as "marked" | "market"));
  const executionRequest = (leaf: number, account: { equity: number; price: number; exposure: number }) => {
    const request = execution![leaf](account), before = account.exposure * account.equity / account.price, after = before + request.quantity;
    const turnover = Math.abs(request.quantity) * account.price;
    return { ...account, ...request, turnover, cost: turnover * (p.costs.feeBps + p.costs.slippageBps) / 10000,
      accountConvention: "before-order", notionalConvention: "decision-price-diagnostic",
      longEntry: Math.max(0, after) > Math.max(0, before) + 1e-12,
      longExit: Math.max(0, after) < Math.max(0, before) - 1e-12,
      shortEntry: Math.min(0, after) < Math.min(0, before) - 1e-12,
      shortExit: Math.min(0, after) > Math.min(0, before) + 1e-12 };
  };
  const marketClosures = options.marketClosures ?? [];
  validateEventMarketClosures(marketClosures);
  if (marketClosures.length && (interval !== 1000 || options.updates || options.twoStep?.countdown
    || !(options.oneStepTerminal || options.twoStep || options.threeStep)))
    throw new Error("Market availability requires frozen native event planning");
  if (c.some(row => Boolean(row.carriedMark) !== eventMarketUnavailable(marketClosures, row.openTime)))
    throw new Error("Replay marks do not match the declared market availability");
  if (interval === 1000 && (p.model.hidden || options.sign || options.sizeSign || options.severity || options.fitted || options.adaptive))
    throw new Error("Native second replay does not accept minute-specific forecast adapters");
  if (options.updates && options.adaptive) throw new Error("Use one event-policy update mechanism");
  if (options.oneStepTerminal && (depth !== 1 || options.fitted || options.sign || options.sizeSign || options.severity
    || options.updates || options.adaptive || p.model.hidden || options.twoStep || options.threeStep)) throw new Error("Exact one-event replay requires a frozen joint kernel and depth one");
  if (options.twoStep && (depth !== 2 || options.fitted || options.sign || options.sizeSign || options.severity
    || options.updates || options.adaptive || p.model.hidden || options.threeStep)) throw new Error("Bounded two-event replay requires a frozen joint kernel and depth two");
  if (options.threeStep && (depth !== 3 || options.fitted || options.sign || options.sizeSign || options.severity
    || options.updates || options.adaptive || p.model.hidden)) throw new Error("Bounded three-event replay requires a frozen joint kernel and depth three");
  if (options.twoStep?.initialDepth !== undefined && (!options.twoStep.countdown || ![1, 2].includes(options.twoStep.initialDepth)))
    throw new Error("Initial two-event depth requires a countdown controller");
  if (options.fitted && (options.sign || options.sizeSign || options.severity || options.updates || options.adaptive || p.model.hidden
    || JSON.stringify(options.fitted.policy.costs) !== JSON.stringify(p.costs) || options.fitted.policy.limits.length !== p.model.kernels.length))
    throw new Error("Incompatible fitted-value replay");
  if (options.fitted?.replanEvery !== undefined && (options.fitted.replanEvery !== depth || !Number.isInteger(depth) || depth < 1
    || (options.fitted.policy.targetMode !== "minimum-turnover" && !(options.fitted.policy.targetMode === "sampled-path" && depth === 2))))
    throw new Error("Invalid holding-option replay");
  if (options.fitted?.sampledAlternative && (options.fitted.replanEvery !== 2 || depth !== 2
    || options.fitted.policy.targetMode !== "minimum-turnover"
    || options.fitted.sampledAlternative.targetMode !== "sampled-path"
    || JSON.stringify(options.fitted.sampledAlternative.tables[0]) !== JSON.stringify(options.fitted.policy.tables[0])))
    throw new Error("Invalid sampled option alternative");
  if (options.fitted?.replanCash && !options.fitted.replanEvery) throw new Error("Cash replanning requires an option controller");
  const sampledContinuation = options.fitted?.sampledAlternative
    ?? (options.fitted?.replanEvery && options.fitted.policy.targetMode === "sampled-path" ? options.fitted.policy : undefined);
  const initialState = options.fitted?.initialState;
  if (initialState && (!options.fitted?.replanEvery || !Number.isInteger(initialState.remaining) || initialState.remaining < 0
    || initialState.remaining > options.fitted.replanEvery || ![0, 1].includes(initialState.controller)
    || (initialState.controller === 1 && !sampledContinuation))) throw new Error("Invalid initial option state");
  if (options.sign?.head.objective === "weighted-sign") throw new Error("Return-weighted scores require conditional-probability inversion before replay");
  if (options.sign && (options.updates || options.adaptive || p.model.hidden || options.sign.lookahead.policy !== p
    || !(options.sign.blend >= 0 && options.sign.blend <= 1))) throw new Error("Incompatible sign lookahead replay");
  if (options.sizeSign && (options.sign || options.updates || options.adaptive || p.model.hidden || options.sizeSign.lookahead.policy !== p
    || !(options.sizeSign.blend >= 0 && options.sizeSign.blend <= 1)
    || (options.sizeSign.observations && (options.sizeSign.fastVolatility || options.sizeSign.eventHistory)))) throw new Error("Incompatible size/sign lookahead replay");
  if (options.severity && (!options.sizeSign || options.severity.lookahead.policy !== p
    || options.severity.head.thresholdLogBps !== options.sizeSign.head.thresholdLogBps || !(options.severity.blend >= 0 && options.severity.blend <= 1)))
    throw new Error("Severity replay requires a compatible frozen size/sign head and kernel lookahead");
  const contract = (policy: EventPolicy) => JSON.stringify([policy.costs, policy.equities, policy.prices, policy.exposures, policy.targets,
    policy.model.clock, policy.model.featureNames, policy.model.nodes, policy.model.forest, policy.model.projection, policy.model.boost,
    policy.model.hidden, policy.model.runSymmetry, policy.model.runConditioned, policy.model.runVolatility]);
  if (options.updates?.some((u, i, updates) => !Number.isFinite(u.at) || u.at < start || u.at >= end
    || (i > 0 && u.at <= updates[i - 1].at) || contract(u.policy) !== contract(p) || u.policy.tables.length < depth))
    throw new Error("Policy updates must preserve account/state contracts and have ordered in-range availability");
  let updateIndex = 0, activePolicy = p, policyUpdatedAt: number | undefined;
  let optionRemaining = initialState?.remaining ?? 0;
  let twoStepDepth = options.twoStep?.initialDepth ?? 2;
  const twoStep = options.twoStep ? prepareEventTwoStep(p.model, p.costs, options.twoStep.terminal,
    { globalUpper: options.twoStep.globalUpper }) : undefined;
  const threeStep = options.threeStep ? prepareEventThreeStep(p.model, p.costs, options.threeStep.terminal, options.threeStep) : undefined;
  let optionController = initialState?.controller ?? (options.fitted?.sampledAlternative ? 0 : sampledContinuation ? 1 : 0);
  const eventHistory = options.sizeSign?.eventHistory ? new EventCompletedHistory(start) : undefined;
  const initial = options.equity ?? 10_000;
  let equity = initial, quantity = options.initialQuantity ?? 0, peak = initial, maxDrawdown = 0, fees = 0, borrow = 0;
  if (!Number.isFinite(initial) || initial <= 0 || !Number.isFinite(quantity)) throw new Error("Invalid initial replay account");
  let closePeak = initial, closeDrawdown = 0;
  let trades = 0, reversals = 0, liquidations = 0, canceledOrders = 0, decisions = 0, exposedMinutes = 0;
  let longPnl = 0, shortPnl = 0, longMinutes = 0, shortMinutes = 0, predictedGain = 0;
  let gridBoundaryDecisions = 0;
  let unavailableSeconds = 0, unavailableBorrow = 0, unavailableCanceledOrders = 0;
  let availabilityScoredUntil = start;
  const forcedWaits: Array<{ time: number; endTime: number; equityBefore: number; equityAfter: number;
    quantityBefore: number; quantityAfter: number; borrowing: number; positionChanges: unknown[] }> = [];
  const adaptationPairs = options.adaptive?.initialPairs?.slice(-options.adaptive.window) ?? [];
  let scaleSum = 0, scaleDecisions = 0;
  let firstEquity = initial, day = "", lastPrice = 0;
  const daily: { date: string; logReturn: number }[] = [], trace: Record<string, unknown>[] = [];
  let i = c.findIndex((v, j) => j >= warmup && v.openTime + interval >= start);
  if (i < warmup) throw new Error("No replay warmup");
  const positions = (options.positionAttribution ?? options.trace ?? false)
    ? new EventPositionLedger(options.trace ?? false) : undefined;
  lastPrice = c[i].close;
  if (quantity) positions?.fill({ time: c[i].openTime + interval, price: lastPrice, quantityAfter: quantity, cost: 0, origin: "initial-mark" });
  const lastReplayIndex = c.findLastIndex(v => v.openTime < end);
  if (lastReplayIndex < 0 || c[lastReplayIndex].openTime + interval < end) throw new Error("Incomplete replay history at boundary");
  const observedCandles = lastReplayIndex === c.length - 1 ? c : c.slice(0, lastReplayIndex + 1);
  let belief = p.model.hidden?.initial ?? 0;
  if (p.model.hidden) {
    const past = c.slice(0, i + 1);
    for (let j = Math.max(warmup, i - warmup); j < i;) {
      const observed = observeMove(past, j, p.model.clock, p.model.featureNames);
      if (!observed) break;
      belief = eventHiddenNext(p.model, belief, observed.label); j = observed.end;
    }
  }
  for (; i + 1 < c.length && c[i + 1].openTime < end;) {
    if (c[i].openTime - c[i - warmup].openTime !== warmup * interval) throw new Error("Gap in replay warmup");
    const time = c[i].openTime + interval, current = c[i].close;
    // The last completed second reveals current unavailability. No scheduled
    // reopening time is supplied to the decision model or its action search.
    const forcedWait = Boolean(c[i].carriedMark);
    const move = forcedWait ? null : observeMove(observedCandles, i, p.model.clock, p.model.featureNames);
    // The final event may be censored by the scoring boundary. Price history
    // through that boundary is enough to execute it and settle, without reading
    // the next evaluation window to discover its eventual stopping time.
    if (!forcedWait && !move && observeMove(observedCandles, i, p.model.clock, eventBaseFeatures(p.model.clock)))
      throw new Error(`Missing replay event features at ${new Date(time).toISOString()}`);
    const features = forcedWait || p.model.hidden ? undefined : eventFeatures(c, i, p.model.featureNames, p.model.clock);
    const leaf = forcedWait ? -1 : p.model.hidden ? belief : eventLeaf(p.model, features!);
    const account = { equity, price: current, exposure: quantity * current / equity };
    const positionExitDomains = options.trace && !forcedWait ? positions?.closeDomains() : undefined;
    const positionChanges: Array<{ time: number; price: number; reason: string;
      changes: ReturnType<EventPositionLedger["fill"]> }> = [];
    if (!forcedWait && (equity < p.equities[0] || equity > p.equities.at(-1)! || current < p.prices[0] || current > p.prices.at(-1)!
      || account.exposure < p.exposures[0] || account.exposure > p.exposures.at(-1)!)) gridBoundaryDecisions++;
    while (options.updates && updateIndex < options.updates.length && options.updates[updateIndex].at <= time) {
      const update = options.updates[updateIndex++]; activePolicy = update.policy; policyUpdatedAt = update.at;
    }
    let decisionPolicy = activePolicy, forecastScale = 1;
    if (options.adaptive) {
      forecastScale = rollingEventScale(adaptationPairs, options.adaptive.window);
      const index = options.adaptive.scales.reduce((best, value, j) =>
        Math.abs(value - forecastScale) < Math.abs(options.adaptive!.scales[best] - forecastScale) ? j : best, 0);
      decisionPolicy = options.adaptive.policies[index]; forecastScale = options.adaptive.scales[index];
    }
    const baseSign = options.sign ? eventSignMass(p.model.kernels[leaf]).probability : undefined;
    const observations = options.sign?.observations ?? options.sizeSign?.observations ?? options.fitted?.observations;
    const signObservation = observations?.get(time);
    if (observations && (!signObservation || signObservation.availableAt !== time))
      throw new Error("Missing or misaligned completed sign observation");
    const signProbability = options.sign ? (1 - options.sign.blend) * baseSign!
      + options.sign.blend * predictEventSign(options.sign.head, signObservation?.values ?? features!) : undefined;
    const eventHistoryFeatures = eventHistory?.features(time);
    let sizeSignProbabilities = options.sizeSign ? predictEventSizeSigns(options.sizeSign.head,
      signObservation?.values ?? [...features!, ...(options.sizeSign.fastVolatility ? eventFastVolatilityFeatures(c, i) : []), ...(eventHistoryFeatures ?? [])]) : undefined;
    if (options.sizeSign?.components) sizeSignProbabilities = selectEventSizeSignComponents(options.sizeSign.lookahead.masses[leaf], sizeSignProbabilities!, options.sizeSign.components);
    const sizeSignMass = options.sizeSign ? mixEventSizeSigns(options.sizeSign.lookahead.masses[leaf], sizeSignProbabilities!, options.sizeSign.blend) : undefined;
    const severityTargets = options.severity ? predictEventSeverity(options.severity.head, Math.expm1(eventFastVolatilityFeatures(c, i)[1])) : undefined;
    const severityKernel = options.severity ? tiltEventSeverity(reweightEventSizeSigns(decisionPolicy.model.kernels[leaf],
      options.sizeSign!.head.thresholdLogBps, sizeSignProbabilities!, options.sizeSign!.blend), options.severity.head.thresholdLogBps, severityTargets!, options.severity.blend) : undefined;
    const optionHolding = Boolean(options.fitted?.replanEvery && optionRemaining > 0
      && (account.exposure !== 0 || (optionController === 1 && !options.fitted.replanCash)));
    const composition = options.fitted?.sampledAlternative && !optionHolding ? decideFittedEventControllers(
      [options.fitted.policy, options.fitted.sampledAlternative], signObservation!.values, account, depth, leaf) : undefined;
    const decision = forcedWait ? undefined : execution ? executionRequest(leaf, account)
      : options.oneStepTerminal ? decideEventOneStep(decisionPolicy.model.kernels[leaf], account, p.costs, options.oneStepTerminal)
      : options.twoStep ? twoStepDepth === 1
        ? decideEventOneStep(decisionPolicy.model.kernels[leaf], account, p.costs, options.twoStep.terminal)
        : twoStep!(leaf, account, options.twoStep)
      : options.threeStep ? threeStep!(leaf, account, options.threeStep)
      : optionHolding ? optionController === 1
      ? decideFittedEvent(sampledContinuation!, signObservation!.values, account, 1, leaf) : chooseEventTrade(p, account, () => 0)
      : composition ? composition.decision
      : options.fitted ? decideFittedEvent(options.fitted.policy, signObservation!.values, account, depth, leaf)
      : options.severity ? decideEventKernel(options.severity.lookahead, leaf, account, depth, severityKernel!)
      : options.sizeSign ? decideEventOutcomes(options.sizeSign.lookahead, leaf, account, depth, sizeSignMass!)
      : options.sign ? decideEventSign(options.sign.lookahead, leaf, account, depth, signProbability!)
      : decideEvent(decisionPolicy, leaf, account, depth);
    if (options.fitted?.replanEvery && !optionHolding) {
      optionRemaining = options.fitted.replanEvery; optionController = composition?.controller ?? (sampledContinuation ? 1 : 0);
    }
    if (decision) { scaleSum += forecastScale; scaleDecisions++; decisions++; predictedGain += decision.value; }
    let dq = options.cash ? 0 : decision?.quantity ?? 0;
    const quoteOrderQty = options.quoteEntries && quantity === 0 && dq ? decision!.turnover : undefined;
    const before = equity, beforeQuantity = quantity, beforeBorrow = borrow;
    let endIndex = move?.end ?? lastReplayIndex;
    if (marketClosures.length) {
      // These are stopping rules in the simulator, not future inputs to a
      // forecast. A wait ends only after the first available second completes.
      for (let j = i + 1; j <= endIndex; j++) if (forcedWait ? !c[j].carriedMark : c[j].carriedMark) { endIndex = j; break; }
    }
    const interruptedByUnavailable = !forcedWait && Boolean(c[endIndex].carriedMark);
    let processedEndIndex = i;
    for (let j = i + 1; j <= endIndex; j++) {
      if (c[j].openTime - c[j - 1].openTime !== interval) throw new Error("Gap in scored history");
      processedEndIndex = j;
      availabilityScoredUntil = c[j].openTime + interval;
      const unavailable = eventMarketUnavailable(marketClosures, c[j].openTime);
      if (unavailable) unavailableSeconds++;
      const label = new Date(c[j].openTime).toISOString().slice(0, 10);
      if (label !== day) {
        if (day) daily.push({ date: day, logReturn: Math.log(equity / firstEquity) });
        firstEquity = equity; day = label;
      }
      // Orders are committed at the close and filled at the NEXT open.
      // Quote entries commit turnover; all other orders commit base quantity.
      const gapPnl = quantity * (c[j].open - c[j - 1].close);
      equity += gapPnl;
      peak = Math.max(peak, equity);
      if (quantity > 0) longPnl += gapPnl; else shortPnl += gapPnl;
      if (equity <= p.costs.maintenanceMargin * Math.abs(quantity) * c[j].open) {
        if (quantity) {
          if (positions) positionChanges.push({ time: c[j].openTime, price: c[j].open, reason: "liquidation",
            changes: positions.liquidate(c[j].openTime, c[j].open, equity) });
          liquidations++; equity = 0; quantity = 0;
        }
      }
      if (equity <= 0) break;
      if (j === i + 1 && dq && unavailable) { canceledOrders++; unavailableCanceledOrders++; dq = 0; }
      if (j === i + 1 && dq) {
        if (quoteOrderQty !== undefined)
          dq = Math.sign(dq) * Math.floor(quoteOrderQty / c[j].open / p.costs.quantityStep + 1e-8) * p.costs.quantityStep;
        const turnover = Math.abs(dq) * c[j].open;
        const cost = turnover * (p.costs.feeBps + p.costs.slippageBps) / 1e4;
        if (Math.abs(dq) < p.costs.minQuantity - 1e-12 || turnover > p.costs.maxNotional + 1e-8 || turnover < p.costs.minNotional - 1e-8
          || equity <= cost || !eventCapAllowsTrade(quantity * c[j].open / equity,
            (quantity + dq) * c[j].open / (equity - cost), turnover, c[j].open, p.costs)) {
          canceledOrders++; dq = 0;
        } else {
          if (quantity * (quantity + dq) < 0) reversals++;
          if (positions) positionChanges.push({ time: c[j].openTime, price: c[j].open, reason: "fill",
            changes: positions.fill({ time: c[j].openTime, price: c[j].open,
              quantityAfter: Math.round((quantity + dq) / p.costs.quantityStep) * p.costs.quantityStep, cost }) });
          quantity = Math.round((quantity + dq) / p.costs.quantityStep) * p.costs.quantityStep;
          equity -= cost; fees += cost; trades++;
        }
      }
      const exposure = quantity * c[j].open / equity;
      const held = eventHolding(exposure, { return: c[j].close / c[j].open - 1,
        low: c[j].low / c[j].open - 1, high: c[j].high / c[j].open - 1, duration: barMinutes }, p.costs);
      if (held.liquidated) {
        if (positions) positionChanges.push({ time: c[j].openTime + interval, price: c[j].open, reason: "liquidation",
          changes: positions.liquidate(c[j].openTime + interval, c[j].open, equity) });
        liquidations++; equity = 0; quantity = 0; break;
      }
      const pnl = quantity * (c[j].close - c[j].open);
      const newEquity = equity * held.factor;
      const periodBorrow = equity + pnl - newEquity;
      borrow += periodBorrow;
      if (unavailable) unavailableBorrow += periodBorrow;
      if (periodBorrow) positions?.charge(c[j].openTime + interval, periodBorrow);
      if (quantity > 0) { longPnl += pnl; longMinutes += barMinutes; }
      else if (quantity < 0) { shortPnl += pnl; shortMinutes += barMinutes; }
      if (quantity) exposedMinutes += barMinutes;
      // OHLC does not reveal high/low order. Report the conservative envelope:
      // favorable excursion first, then adverse excursion with full bar borrow.
      const favorableEquity = equity + quantity * ((quantity >= 0 ? c[j].high : c[j].low) - c[j].open);
      const adverseEquity = equity + quantity * ((quantity >= 0 ? c[j].low : c[j].high) - c[j].open) - periodBorrow;
      peak = Math.max(peak, favorableEquity);
      maxDrawdown = Math.max(maxDrawdown, 1 - adverseEquity / peak);
      equity = newEquity; peak = Math.max(peak, equity); lastPrice = c[j].close;
      maxDrawdown = Math.max(maxDrawdown, 1 - equity / peak);
      closePeak = Math.max(closePeak, equity); closeDrawdown = Math.max(closeDrawdown, 1 - equity / closePeak);
    }
    if (forcedWait) forcedWaits.push({ time, endTime: c[processedEndIndex].openTime + interval,
      equityBefore: before, equityAfter: equity, quantityBefore: beforeQuantity, quantityAfter: quantity,
      borrowing: borrow - beforeBorrow, positionChanges });
    if (decision && (options.trace || options.onDecision)) {
      const traceEndIndex = marketClosures.length ? processedEndIndex : endIndex;
      const row = { time, endTime: c[traceEndIndex].openTime + interval, leaf,
      ...(interruptedByUnavailable ? { interruptedByUnavailable: true } : {}),
      ...(options.oneStepTerminal ? { optimizer: execution ? "execution-one-event-lots" : "one-event-lots", terminal: options.oneStepTerminal } : {}),
      ...(options.twoStep ? { optimizer: twoStepDepth === 2 ? "two-event-bounds" : "one-event-lots", terminal: options.twoStep.terminal } : {}),
      ...(options.twoStep?.countdown ? { decisionDepth: twoStepDepth } : {}),
      ...(options.threeStep ? { optimizer: "three-event-bounds", terminal: options.threeStep.terminal } : {}),
      ...(quoteOrderQty !== undefined ? { quoteOrderQty } : {}),
      ...(options.fitted ? { forecastSource: "original-distribution-diagnostic", fittedValueDepth: depth } : {}),
      ...(options.fitted?.replanEvery ? { optionRemaining, optionHolding } : {}),
      ...(sampledContinuation ? { optionController: optionController === 0 ? "hold" : "sampled" } : {}),
      expectedReturnBps: (severityKernel ?? (options.sizeSign ? reweightEventSizeSigns(decisionPolicy.model.kernels[leaf], options.sizeSign.head.thresholdLogBps,
        sizeSignProbabilities!, options.sizeSign.blend) : options.sign && signProbability! > 0 && signProbability! < 1
        ? reweightEventSigns(decisionPolicy.model.kernels[leaf], signProbability!) : decisionPolicy.model.kernels[leaf]))
        .reduce((s, a) => s + a.probability * a.return, 0) * 1e4,
      rawExpectedReturnBps: p.model.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0) * 1e4,
      forecastScale, policyUpdatedAt, signProbability, sizeSignProbabilities, eventHistoryFeatures,
      severityTargets, achievedSeverity: severityKernel ? eventSeverityMeans(severityKernel, options.severity!.head.thresholdLogBps) : undefined,
      realizedReturnBps: (c[traceEndIndex].close / current - 1) * 1e4, equityBefore: before, equityAfter: equity,
      exposureBefore: account.exposure, exposureAfter: quantity * c[endIndex].close / Math.max(1e-12, equity),
      orderQuantity: dq, previousQuantity: beforeQuantity, order: decision,
      ...(positions ? { positionExitDomains, positionChanges, positionQuantity: positions.quantity() } : {}) };
      if (options.trace) trace.push(row);
      options.onDecision?.(row);
    }
    if (options.adaptive && move) {
      const raw = p.model.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0);
      adaptationPairs.push([raw, c[endIndex].close / current - 1]);
      if (adaptationPairs.length > options.adaptive.window) adaptationPairs.shift();
    }
    if (p.model.hidden && move) belief = eventHiddenNext(p.model, belief, move.label);
    if (eventHistory && move) eventHistory.observe({ originTime: time, availableAt: c[move.end].openTime + interval,
      return: move.return, duration: move.duration }, c[move.end].openTime + interval);
    if (equity <= 0) { maxDrawdown = 1; closeDrawdown = 1; break; }
    if (options.fitted?.replanEvery) optionRemaining = Math.max(0, optionRemaining - 1);
    if (options.twoStep?.countdown) twoStepDepth = twoStepDepth === 2 ? 1 : 2;
    i = endIndex;
  }
  // Terminal cash settlement pays the same friction on every remaining unit.
  // Large closes are split into maxNotional market clips; sub-minimum dust is marked.
  // A scheduled reopening without a completed available price cannot justify
  // selling at the carried mark. Keep the account unsettled at that boundary.
  const terminalUnavailable = eventMarketUnavailable(marketClosures, end) || Boolean(c[lastReplayIndex].carriedMark);
  const unsettledNotional = terminalUnavailable ? Math.abs(quantity) * lastPrice : 0;
  const settlement = terminalUnavailable ? { equity, quantity, fee: 0, orders: 0, dust: 0 }
    : settleEventReplayAccount(equity, quantity, lastPrice, p.costs);
  const terminalPositionChanges = positions && settlement.quantity !== quantity
    ? positions.fill({ time: end, price: lastPrice, quantityAfter: settlement.quantity, cost: settlement.fee, reason: "terminal" }) : [];
  equity = settlement.equity; quantity = settlement.quantity; fees += settlement.fee; trades += settlement.orders;
  const terminalDust = settlement.dust;
  maxDrawdown = Math.max(maxDrawdown, 1 - equity / peak);
  closeDrawdown = Math.max(closeDrawdown, 1 - equity / closePeak);
  if (day) daily.push({ date: day, logReturn: equity > 0 ? Math.log(equity / firstEquity) : -100 });
  const positionSummary = positions?.summary(lastPrice);
  if (positionSummary && (Math.abs(positionSummary.collapsedQuantity - quantity) > 1e-10 * Math.max(1, Math.abs(quantity))
    || Math.abs(initial + positionSummary.equityChange - equity) > 1e-7 + 1e-10 * Math.max(initial, Math.abs(equity))))
    throw new Error("Lifecycle positions do not reconcile with the replay account");
  return { returnPct: (equity / initial - 1) * 100, logGrowth: equity > 0 ? Math.log(equity / initial) : -100,
    finalEquity: equity, maxDrawdownPct: maxDrawdown * 100, closeDrawdownPct: closeDrawdown * 100, fees, borrow, trades, reversals,
    liquidations, canceledOrders, decisions, exposedMinutes, longMinutes, shortMinutes,
    longPnl, shortPnl, predictedGain, terminalDust, gridBoundaryDecisions,
    meanForecastScale: scaleDecisions ? scaleSum / scaleDecisions : 1, adaptationPairs, daily, trace,
    ...(marketClosures.length ? { marketAvailability: { unavailableSeconds, unavailableBorrow, unavailableCanceledOrders,
      forcedWaits, terminalUnavailable, unsettledNotional, terminalQuantity: quantity, scoredUntil: availabilityScoredUntil } } : {}),
    ...(positionSummary ? { positionSummary } : {}),
    ...(positions && options.trace ? { positions: positions.positions(), terminalPositionChanges } : {}) };
}

function argument(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`); return i < 0 ? fallback : process.argv[i + 1];
}
export function loadEventCandles(start: number, end: number, intervalMs: 1000 | 60000 = 60000,
  marketClosures?: readonly EventMarketClosure[]): EventCandle[] {
  if (![1000, 60000].includes(intervalMs) || !Number.isSafeInteger(start) || !Number.isSafeInteger(end)
    || end <= start || start % intervalMs || end % intervalMs) throw new Error("Invalid candle source interval or boundaries");
  if (marketClosures) { validateEventMarketClosures(marketClosures); if (intervalMs !== 1000) throw new Error("Availability views require native seconds"); }
  const sourceStart = eventAvailabilitySourceStart(start, marketClosures ?? []);
  const c: EventSourceSecond[] = [];
  const dir = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt", intervalMs === 1000 ? "1s" : "1m");
  for (let day = Math.floor(sourceStart / DAY) * DAY; day < end; day += DAY) {
    const file = path.join(dir, `${new Date(day).toISOString().slice(0, 10)}.json`);
    if (!fs.existsSync(file)) {
      if (intervalMs === 1000) throw new Error(`Missing native second shard: ${file}`);
      continue;
    }
    for (const candle of readCandleShardReferenceSync(file)) {
      if (intervalMs === 1000) {
        if (candle.openTime < sourceStart || candle.openTime >= end) continue;
        if (!marketClosures) validateEventSourceSecond(candle);
      }
      c.push(candle);
    }
  }
  if (marketClosures) return markUnavailableEventSeconds(c, start, end, marketClosures);
  if (!c.length) throw new Error("No cached event candles");
  for (let i = 1; i < c.length; i++) if (c[i].openTime <= c[i - 1].openTime) throw new Error("Unordered/duplicate candles");
  if (intervalMs === 1000 && (c.length !== (end - start) / 1000 || c.some((row, i) => row.openTime !== start + i * 1000)))
    throw new Error("Gap in native second candle source");
  return c;
}

/** Validate each native source block in full. Deliberate gaps between blocks
 * remain gaps; feature and outcome timestamp guards reject crossing them. */
export function loadNativeEventCandles(ranges: readonly EventSourceRange[]): EventCandle[] {
  const candles: EventCandle[] = [];
  for (const range of mergeEventSourceRanges(ranges))
    for (const row of loadEventCandles(range.start, range.end, 1000)) candles.push(row);
  if (!candles.length) throw new Error("No native event source ranges");
  return candles;
}

export function eventCalibrationRanges(startTime: number, endTime: number, excluded: readonly Range[]): Range[] {
  const ranges: Range[] = [];
  for (let start = startTime; start < endTime;) {
    const overlap = excluded.find(w => start < w.endTime + DAY && start >= w.startTime - DAY);
    if (overlap) { start = overlap.endTime + DAY; continue; }
    const next = Math.min(endTime, ...excluded.filter(w => w.startTime - DAY > start).map(w => w.startTime - DAY));
    if (next - start >= DAY) ranges.push({ id: "cal", startTime: start, endTime: next });
    start = next;
  }
  return ranges;
}

export function rollingEventScale(pairs: readonly [number, number][], window: number): number {
  const completed = pairs.slice(-window);
  if (completed.length < 4) return 1;
  const xx = completed.reduce((s, [x]) => s + x * x, 0), xy = completed.reduce((s, [x, y]) => s + x * y, 0);
  return xx > 1e-15 ? Math.max(-2, Math.min(2, xy / xx)) : 0;
}

async function main() {
  const all = new KamaInspector(path.join(root, "data")).catalog().windows.filter(w => w.id !== "latest" && !w.id.startsWith("fit-"));
  const selected = argument("windows", "sideways-churn-2021-09,regime-down-2022-06,sharpe-up-3d-2024-11");
  if (selected !== "all" && selected.split(",").some(id => !all.some(w => w.id === id))) throw new Error("Unknown or excluded inspector window");
  const windows = selected === "all" ? all : all.filter(w => selected.split(",").includes(w.id));
  if (!windows.length) throw new Error("No matching inspector windows");
  const clock = { thresholdBps: Number(argument("threshold-bps", "20")), maxCandles: Number(argument("max-candles", "60")),
    runClock: process.argv.includes("--run-clock"), reversalClock: process.argv.includes("--reversal-clock"),
    progressBps: process.argv.includes("--progress-bps") ? Number(argument("progress-bps", "0")) : undefined };
  const trainDays = Number(argument("train-days", "120")), calibrationDays = Number(argument("calibration-days", "21"));
  const stride = Number(argument("stride", "10")), maxDepth = Number(argument("bellman-depth", "4"));
  const costs = { ...DEFAULT_EVENT_COSTS, feeBps: Number(argument("fee-bps", "10")),
    slippageBps: Number(argument("slippage-bps", "2")), maxLeverage: Number(argument("leverage", "5")) };
  const treeDepths = argument("tree-depths", "0,2,4").split(",").map(Number);
  const prior = Number(argument("prior", "500"));
  const minLeaf = Number(argument("min-leaf", "100"));
  const riskPenalty = Number(argument("risk-penalty", "0.1"));
  if (!Number.isFinite(riskPenalty) || riskPenalty < 0) throw new Error("Invalid risk penalty");
  const honestyFraction = Number(argument("honesty-fraction", "0"));
  const forestSize = Number(argument("forest-size", "0"));
  const learner = argument("learner", "tree");
  if (!["tree", "projection", "boost", "hidden"].includes(learner) || (learner !== "tree" && forestSize)
    || (["boost", "hidden"].includes(learner) && honestyFraction)) throw new Error("Invalid learner/forest/honesty combination");
  const ridgePenalties = argument("ridge-penalties", "0.01,0.1,1").split(",").map(Number);
  const projectionCells = Number(argument("projection-cells", "8"));
  const boostIterations = argument("boost-iterations", "16,64,256").split(",").map(Number);
  const boostRate = Number(argument("boost-rate", "0.05"));
  const hiddenStates = argument("hidden-states", "2,3").split(",").map(Number);
  const hiddenResolution = Number(argument("hidden-resolution", "4"));
  const hiddenSmoothing = Number(argument("hidden-smoothing", "10"));
  const trainingIsolation = argument("training-isolation", "suite");
  if (!["suite", "causal"].includes(trainingIsolation)) throw new Error("Unknown training isolation protocol");
  const sampling = argument("sampling", "stride") as "stride" | "chain";
  if (!["stride", "chain"].includes(sampling)) throw new Error("Unknown sampling mode");
  const criterion = argument("criterion", "distribution") as "distribution" | "mean";
  if (!["distribution", "mean"].includes(criterion)) throw new Error("Unknown tree criterion");
  const modelSelection = argument("model-selection", "prediction");
  if (!["prediction", "utility"].includes(modelSelection)) throw new Error("Unknown model selection objective");
  const secondBasis = process.argv.includes("--second-basis")
    ? new EventSecondBasis(path.join(root, "data/runtime-cache/global-feature-basis")) : undefined;
  if (secondBasis && process.argv.includes("--invert-augment")) throw new Error("Second-basis inversion requires rematerialized second candles");
  if (secondBasis && process.argv.includes("--path-basis")) throw new Error("Screen second and path bases separately");
  const runBasis = process.argv.includes("--run-basis") || clock.reversalClock;
  if (runBasis && (secondBasis || process.argv.includes("--path-basis"))) throw new Error("Screen run, second and path bases separately");
  const featureNames = runBasis ? EVENT_RUN_FEATURES : secondBasis ? EVENT_SECOND_FEATURES : process.argv.includes("--path-basis") ? EVENT_PATH_FEATURES : EVENT_FEATURES;
  const codeHash = createHash("sha256");
  const sourceFiles = ["scripts/research-event-policy.ts", "scripts/event-second-basis.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-hidden.ts", "packages/bot-algo/src/event-run-model.ts", "packages/bot-algo/src/event-log-policy.ts"];
  for (const file of sourceFiles)
    codeHash.update(fs.readFileSync(path.join(root, file)));
  const config = { clock, trainDays, calibrationDays, stride, maxDepth, costs, treeDepths, prior, minLeaf, riskPenalty, honestyFraction, forestSize, criterion,
    learner, ridgePenalties, projectionCells, boostIterations, boostRate, trainingIsolation, hiddenStates, hiddenResolution, hiddenSmoothing, modelSelection,
    riskMarking: "worst-order intrabar OHLC drawdown envelope; separate close-only drawdown",
    featureNames, secondBasisFingerprint: secondBasis?.fingerprint,
    sampling, invertAugment: process.argv.includes("--invert-augment"), calibrateMean: process.argv.includes("--calibrate-mean"),
    onlineScale: process.argv.includes("--online-scale"),
    runSymmetry: process.argv.includes("--run-symmetry"),
    runDirectionPrior: process.argv.includes("--run-direction-prior") ? Number(argument("run-direction-prior", "0")) : undefined,
    refitLatest: process.argv.includes("--refit-latest"),
    actionSteps: Number(argument("action-steps", "5")), sourceHash: codeHash.digest("hex"), windows: windows.map(w => w.id) };
  if (config.refitLatest && (config.onlineScale || config.calibrateMean || (config.invertAugment && learner !== "hidden")))
    throw new Error("Latest refit requires fixed calibration; inverse refit is supported for hidden sequences");
  if (learner === "hidden" && (sampling !== "chain" || config.onlineScale || config.calibrateMean
    || featureNames !== EVENT_FEATURES)) throw new Error("Hidden learner requires event chains and its own unscaled belief state");
  if (config.runDirectionPrior !== undefined && (config.runSymmetry || !Number.isFinite(config.runDirectionPrior) || config.runDirectionPrior < 0))
    throw new Error("Run direction prior must be finite, nonnegative and selected instead of exact symmetry");
  if ((config.runSymmetry || config.runDirectionPrior !== undefined) && (learner !== "tree" || forestSize || !clock.reversalClock || !runBasis
    || config.invertAugment || config.calibrateMean || config.onlineScale))
    throw new Error("Canonical run models require a reversal-clock tree without separate mirroring or mean tilts");
  const id = argument("output", `event-policy-${createHash("sha256").update(JSON.stringify(config)).digest("hex").slice(0, 10)}`);
  const out = path.resolve(root, "data/benchmarks", id);
  fs.mkdirSync(out, { recursive: true });
  const oldConfig = path.join(out, "config.json");
  if (fs.existsSync(oldConfig)) {
    const previous = JSON.parse(fs.readFileSync(oldConfig, "utf8"));
    if (Object.entries(config).some(([k, v]) => JSON.stringify(previous[k]) !== JSON.stringify(v)))
      throw new Error("Source or configuration changed: choose a new --output name instead of reusing stale results");
  }
  fs.writeFileSync(path.join(out, "sources.json"), JSON.stringify(Object.fromEntries(sourceFiles.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
  fs.writeFileSync(path.join(out, "config.json"), JSON.stringify({ ...config, excludedWindows: all,
    contract: "causal-event-tree-bellman-v1", execution: "closed-candle signal; next-open market order; every-minute OHLC risk",
    selection: `${modelSelection === "utility" ? "joint model and depth selection by" : `${criterion === "mean" ? "expectation MSE" : "distribution NLL"} then`} log-growth minus ${riskPenalty} drawdown on preceding calibration; cash eligible`,
    caveat: "Inspector windows are a repeatedly inspected research suite, not an untouched final holdout" }, null, 2));
  const results: Record<string, unknown>[] = [];
  for (const window of windows) {
    const resultFile = path.join(out, `${window.id}.json`);
    if (fs.existsSync(resultFile)) { results.push(JSON.parse(fs.readFileSync(resultFile, "utf8"))); continue; }
    const started = performance.now(), calibrationStart = window.startTime - calibrationDays * DAY;
    const trainStart = calibrationStart - trainDays * DAY;
    const trainingExclusions = trainingIsolation === "causal" ? [window] : all;
    const selectionExcludedWindows = trainingIsolation === "causal" ? [] : all;
    console.log(JSON.stringify({ event: "load", window: window.id, start: new Date(trainStart).toISOString() }));
    const candles = loadEventCandles(trainStart - 2 * DAY, window.endTime + DAY);
    secondBasis?.attach(candles);
    const trainingSamples = (start: number, end: number) => {
      const rows = makeSamples(candles, clock, start, end, trainingExclusions, stride, sampling, featureNames);
      if (config.invertAugment) {
        const mirrored = makeSamples(invertEventCandles(candles), clock, start, end, trainingExclusions, stride, sampling, featureNames);
        rows.push(...mirrored.map(s => ({ ...s, series: 1 })));
        if (learner !== "hidden") rows.sort((a, b) => a.start - b.start || a.end - b.end);
      }
      return rows;
    };
    const fit = trainingSamples(trainStart, calibrationStart - DAY);
    const calibration = makeSamples(candles, clock, calibrationStart, window.startTime, trainingExclusions, stride, sampling, featureNames);
    const minimumFit = Math.max(200, minLeaf * 2) * (config.invertAugment ? 2 : 1), minimumCalibration = sampling === "chain" ? 25 : 100;
    if (fit.length < minimumFit || !calibration.length) throw new Error(`Insufficient purged data: ${window.id} fit=${fit.length} cal=${calibration.length}`);
    const insufficientCalibration = calibration.length < minimumCalibration;
    const midpoint = calibrationStart + Math.floor(calibrationDays / 2) * DAY;
    const calibrationFit = calibration.filter(s => candles[s.end].openTime + 60_000 < midpoint);
    const calibrationValidation = calibration.filter(s => candles[s.start].openTime + 60_000 >= midpoint);
    const calibrate = config.calibrateMean && calibrationFit.length >= 100 && calibrationValidation.length >= 100;
    const selectionSamples = calibrate ? calibrationValidation : calibration;
    const policyCalibrationStart = calibrate ? midpoint : calibrationStart;
    const specifications = learner === "hidden" ? hiddenStates.map(depth => ({ depth, penalty: undefined, iterations: undefined }))
      : learner === "projection" ? ridgePenalties.map(penalty => ({ depth: 0, penalty, iterations: undefined }))
      : learner === "boost" ? boostIterations.map(iterations => ({ depth: 2, penalty: undefined, iterations }))
      : treeDepths.map(depth => ({ depth, penalty: undefined, iterations: undefined }));
    const learn = (rows: readonly MoveSample[], { depth, penalty, iterations }: { depth: number; penalty?: number; iterations?: number }) => {
      const multiplicity = config.invertAugment ? 2 : 1;
      return config.runSymmetry || config.runDirectionPrior !== undefined
        ? trainEventRunDistribution(rows, clock, { maxDepth: depth, minLeaf, prior, criterion, honestyFraction, directionPrior: config.runDirectionPrior })
        : learner === "hidden" ? trainEventHidden(rows, clock, { states: depth, resolution: hiddenResolution, smoothing: hiddenSmoothing, iterations: 40 })
        : learner === "boost" ? trainEventBoost(rows, clock, { iterations: iterations!, rate: boostRate,
        cells: projectionCells, minLeaf: minLeaf * multiplicity, prior: prior * multiplicity, featureNames })
        : learner === "projection" ? trainEventProjection(rows, clock, { penalty: penalty!, cells: projectionCells,
        minLeaf: minLeaf * multiplicity, prior: prior * multiplicity, honestyFraction, featureNames })
        : forestSize ? trainEventForest(rows, clock, { trees: forestSize, maxDepth: depth,
        minLeaf: minLeaf * multiplicity, prior: prior * multiplicity, seed: 1731, featureNames })
        : trainEventDistribution(rows, clock, { maxDepth: depth, minLeaf: minLeaf * multiplicity,
          prior: prior * multiplicity, criterion, honestyFraction, featureNames });
    };
    const candidates = specifications.map(({ depth, penalty, iterations }) => {
      const raw = learn(fit, { depth, penalty, iterations });
      const model = calibrate ? calibrateEventMean(raw, calibrationFit) : raw;
      return { model, depth, penalty, iterations, validation: distributionMetrics(model, selectionSamples) };
    }).sort((a, b) => insufficientCalibration ? a.depth - b.depth
      : criterion === "mean" ? b.validation.mseSkill - a.validation.mseSkill : a.validation.nll - b.validation.nll);
    console.log(JSON.stringify({ event: "trained", window: window.id, fit: fit.length, calibration: calibration.length,
      models: candidates.map(v => ({ depth: v.depth, penalty: v.penalty, iterations: v.iterations, ...v.validation })), elapsedSec: (performance.now() - started) / 1000 }));
    const referenceIndex = candles.findIndex(v => v.openTime >= calibrationStart);
    const scales = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2];
    // Excluded inspector spans split calibration into disjoint fresh-cash episodes.
    const calibrationRanges = eventCalibrationRanges(policyCalibrationStart, window.startTime, selectionExcludedWindows);
    const adaptationWindows = config.onlineScale ? [8, 32, 128] : [0];
    const screened = candidates.slice(0, modelSelection === "utility" && !insufficientCalibration ? candidates.length : 1).map(candidate => {
      const policy = buildEventPolicy(candidate.model, costs, { depths: maxDepth, referenceEquity: 10_000,
        referencePrice: candles[referenceIndex].open, actionSteps: config.actionSteps });
      const scaledPolicies = config.onlineScale ? scales.map(scale => buildEventPolicy(scaleEventMean(candidate.model, scale), costs,
        { depths: maxDepth, referenceEquity: 10_000, referencePrice: candles[referenceIndex].open, actionSteps: config.actionSteps })) : [];
      const policies = Array.from({ length: maxDepth }, (_, i) => i + 1).flatMap(depth => adaptationWindows.map(adaptationWindow => {
        const metrics = calibrationRanges.filter(r => r.endTime > r.startTime)
          .map(r => replayEventPolicy(candles, policy, r.startTime, r.endTime, depth, adaptationWindow
            ? { adaptive: { policies: scaledPolicies, scales, window: adaptationWindow } } : undefined));
        const logGrowth = metrics.reduce((s, m) => s + m.logGrowth, 0);
        const drawdown = Math.max(0, ...metrics.map(m => m.maxDrawdownPct)) / 100;
        return { depth, adaptationWindow, logGrowth, drawdown, score: metrics.length ? logGrowth - riskPenalty * drawdown : -Infinity,
          trades: metrics.reduce((s, m) => s + m.trades, 0) };
      })).sort((a, b) => b.score - a.score);
      return { candidate, policy, scaledPolicies, policies };
    }).sort((a, b) => b.policies[0].score - a.policies[0].score);
    const { candidate: chosen, scaledPolicies, policies } = screened[0];
    let policy = screened[0].policy;
    const best = policies[0], cash = insufficientCalibration || best.score <= 0;
    const latestCalibration = calibrationRanges.at(-1);
    const initialPairs = best.adaptationWindow && latestCalibration
      ? replayEventPolicy(candles, policy, latestCalibration.startTime, latestCalibration.endTime, best.depth,
        { adaptive: { policies: scaledPolicies, scales, window: best.adaptationWindow } }).adaptationPairs : undefined;
    const selectionPolicy = policy;
    let finalFit = fit, finalTrainStart = trainStart, finalTrainEnd = calibrationStart - DAY;
    if (config.refitLatest) {
      finalTrainStart = window.startTime - trainDays * DAY; finalTrainEnd = window.startTime;
      finalFit = trainingSamples(finalTrainStart, finalTrainEnd);
      if (finalFit.length < minimumFit) throw new Error("Insufficient data for latest refit");
      const model = learn(finalFit, chosen);
      const at = candles.findIndex(c => c.openTime + 60_000 >= window.startTime);
      policy = buildEventPolicy(model, costs, { depths: maxDepth, referenceEquity: 10_000,
        referencePrice: candles[at].close, actionSteps: config.actionSteps });
      console.log(JSON.stringify({ event: "refit", window: window.id, samples: finalFit.length,
        lastTarget: candles[finalFit.reduce((end, s) => Math.max(end, s.end), 0)].openTime + 60_000, testStart: window.startTime }));
    }
    // Freeze selection before looking at any scored inspector candle or label.
    fs.writeFileSync(path.join(out, `${window.id}-model.json`), JSON.stringify({ policy: serializeEventPolicy(policy),
      selectionPolicy: config.refitLatest ? serializeEventPolicy(selectionPolicy) : undefined,
      selectionExcludedWindows,
      chosenDepth: best.depth, cash, costs, trainStart: finalTrainStart, trainEnd: finalTrainEnd,
      selectionTrainingEnd: calibrationStart - DAY,
      calibrationEnd: window.startTime, policyCalibrationStart,
      adaptive: best.adaptationWindow ? { scales, window: best.adaptationWindow, initialPairs } : undefined }));
    const adaptive = best.adaptationWindow ? { policies: scaledPolicies, scales, window: best.adaptationWindow, initialPairs } : undefined;
    const test = replayEventPolicy(candles, policy, window.startTime, window.endTime, best.depth, { trace: true, cash, adaptive });
    const diagnostic = cash ? replayEventPolicy(candles, policy, window.startTime, window.endTime, best.depth, { trace: true, adaptive }) : test;
    fs.writeFileSync(path.join(out, `${window.id}-trades.json`), JSON.stringify(diagnostic.trace));
    const testSamples = makeSamples(candles, clock, window.startTime, window.endTime, [], stride, sampling, featureNames);
    const { trace: _trace, ...testMetrics } = test;
    const { trace: _diagnosticTrace, ...diagnosticMetrics } = diagnostic;
    const result = { window, trainingIsolation, fitSamples: finalFit.length, selectionFitSamples: fit.length, calibrationSamples: calibration.length,
      calibrationApplied: calibrate, meanCalibration: chosen.model.meanCalibration, policyCalibrationStart,
      insufficientCalibration,
      trainingSupport: { firstInput: candles[finalFit.reduce((start, s) => Math.min(start, s.start), Infinity) - 1440].openTime,
        lastTarget: candles[finalFit.reduce((end, s) => Math.max(end, s.end), 0)].openTime + 60_000 },
      learner, runSymmetry: config.runSymmetry, runDirectionPrior: config.runDirectionPrior, hiddenStates: policy.model.hidden?.transition.length,
      treeDepth: chosen.depth, projectionPenalty: chosen.penalty, boostIterations: chosen.iterations, leafCount: policy.model.kernels.length,
      modelCandidates: candidates.map(v => ({ depth: v.depth, penalty: v.penalty, iterations: v.iterations, validation: v.validation,
        policyCalibration: screened.find(s => s.candidate === v)?.policies })),
      policyCalibration: policies, chosenDepth: best.depth, adaptationWindow: best.adaptationWindow, cash, test: testMetrics, diagnosticActive: diagnosticMetrics,
      bellmanConvergence: policy.tables.map(t => ({ depth: t.depth, ...t.convergence })),
      testDistribution: distributionMetrics(policy.model, testSamples), elapsedSec: (performance.now() - started) / 1000 };
    fs.writeFileSync(resultFile, JSON.stringify(result, null, 2)); results.push(result);
    console.log(JSON.stringify({ event: "result", window: window.id, cash, depth: best.depth,
      returnPct: test.returnPct, activeReturnPct: diagnostic.returnPct, trades: diagnostic.trades,
      fees: diagnostic.fees, drawdown: diagnostic.maxDrawdownPct, elapsedSec: result.elapsedSec }));
    fs.writeFileSync(path.join(out, "summary.json"), JSON.stringify(results, null, 2));
  }
  console.log(JSON.stringify({ event: "complete", out, windows: results.length }));
}
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
