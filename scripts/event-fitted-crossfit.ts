import { trainEventFittedValue, type EventValueTransition, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import type { EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventPolicyEvaluationFolds } from "./event-paths.js";

/** Fixed pre-origin time blocks; complete history/target support is purged.
 * Only H1 heads are held out, while common account and ruin guards stay fixed. */
export function fitEventCrossfitContinuations(base: EventPolicy, rows: readonly EventValueTransition[],
  support: readonly { start: number; end: number }[], evaluation: readonly { start: number; end: number }[],
  start: number, end: number, penalty: number, parts: 6 | 20) {
  if (rows.length !== support.length || rows.length !== evaluation.length) throw new Error("Crossfit support count mismatch");
  if (parts !== 6 && parts !== 20) throw new Error("Unsupported crossfit block count");
  const folds = eventPolicyEvaluationFolds(support, evaluation, start, end, parts, 86400000);
  const policies: FittedEventValue[] = new Array(rows.length), models: Array<{ fold: number; policy: FittedEventValue }> = [];
  for (const fold of folds) {
    if (!fold.test.length) continue;
    if (fold.complement.length < 100) throw new Error("Insufficient crossfit training rows");
    const policy = trainEventFittedValue(base, fold.complement.map(i => rows[i]), penalty, 1);
    models.push({ fold: fold.index, policy });
    for (const i of fold.test) { if (policies[i]) throw new Error("Overlapping crossfit blocks"); policies[i] = policy; }
  }
  if (policies.filter(Boolean).length !== rows.length) throw new Error("Incomplete crossfit evaluation support");
  return { policies, folds, models };
}
