import type { EventValueBoostOptions } from "../packages/bot-algo/src/event-value-boost.js";

export interface EventFittedSetting {
  basis: string; penalty: number; boost?: EventValueBoostOptions;
  /** Additional completed paired history required by the matched training cohort. */
  historyMinutes?: 240;
  /** Four exact-minute-boundary dynamics from 64 completed one-second closes. */
  secondDynamics?: true;
  /** Completed spot and futures one-minute close locations within their ranges. */
  candleShape?: true;
  /** Matched cohort with this many complete contiguous future events. */
  pathHorizon?: number;
  /** Evaluate earlier fitted actions on observed paths instead of bootstrapping. */
  sampledPath?: true;
  /** Block-held-out H1 fits for the depth-two sampled continuation target. */
  continuationFolds?: 6 | 20;
}
export function eventFittedSettingName(s: EventFittedSetting): string {
  return `${s.basis}-${s.penalty}` + (s.historyMinutes ? `-m${s.historyMinutes}` : "")
    + (s.secondDynamics ? "-s64" : "")
    + (s.candleShape ? "-shape" : "")
    + (s.pathHorizon ? `-p${s.pathHorizon}` : "") + (s.sampledPath ? "-mc" : "")
    + (s.continuationFolds ? `-cf${s.continuationFolds}` : "")
    + (s.boost ? `-b${s.boost.trees}d${s.boost.depth}n${s.boost.minLeaf}r${s.boost.rate}` : "");
}
