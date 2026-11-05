import { OracleCrraBot, defaultMenu, defaultRiskAversion, finiteActionMenu, type ForecastTree, type MenuConfig, type OracleRules,
  type OracleState, type Rational, type RiskAversion } from "@trading/bot-algo/oracle-crra";
import { rational } from "@trading/bot-algo/oracle-crra";
export { DenseOracleController } from "./oracle-dense-controller.js";

/** Research/paper bot factory. Its adapter must use Oracle's full account transition. */
export function createOracleCrraBot(options: {
  readonly initial: OracleState;
  readonly rules: OracleRules;
  readonly funding: (time: number) => Rational;
  readonly menu?: MenuConfig;
  readonly riskAversion?: RiskAversion;
}): OracleCrraBot {
  return new OracleCrraBot(options.initial, { rules: options.rules,
    riskAversion: options.riskAversion ?? defaultRiskAversion,
    menu: finiteActionMenu(options.menu ?? defaultMenu), funding: options.funding });
}

export interface OraclePriceForecaster {
  predict(input: { readonly observedPrice: number; readonly features: readonly number[] }): Promise<ForecastTree>;
  close(): Promise<void>;
}

/** Serial completed-tick controller using a causal price-distribution provider. */
export class StochasticOracleController {
  private observedTimestamp: number;
  private decidedTimestamp: number | undefined;
  private busy = false;
  private stopped = false;
  constructor(readonly bot: OracleCrraBot, readonly forecaster: OraclePriceForecaster, initialTimestamp: number,
    readonly tickMilliseconds = 1000) {
    if (!Number.isSafeInteger(initialTimestamp) || !Number.isSafeInteger(tickMilliseconds) || tickMilliseconds <= 0)
      throw new Error("Invalid initial timestamp or tick interval");
    this.observedTimestamp = initialTimestamp;
  }

  async onTick(input: { readonly timestamp: number; readonly price: Rational; readonly features: readonly number[] }, certificate = false) {
    if (this.stopped || this.busy) throw new Error("Controller is stopped or processing another tick");
    if (input.timestamp === this.observedTimestamp + this.tickMilliseconds) {
      if (this.decidedTimestamp !== this.observedTimestamp) throw new Error("Previous tick still requires a decision");
      this.bot.observe(input.price);
      this.observedTimestamp = input.timestamp;
    }
    if (input.timestamp !== this.observedTimestamp || this.decidedTimestamp === input.timestamp)
      throw new Error("Ticks must be contiguous and receive exactly one decision");
    if (!rational.eq(input.price, this.bot.state.price)) throw new Error("Price differs from the accepted tick observation");
    this.busy = true;
    try {
      const tree = await this.forecaster.predict({ observedPrice: rational.toNumber(input.price), features: input.features });
      const decision = this.bot.decide(input.price, tree, certificate);
      this.decidedTimestamp = input.timestamp;
      return decision;
    } finally { this.busy = false; }
  }

  /** Observe the final mark, perform the specified valid terminal close, stop inference. */
  async finish(timestamp: number, price: Rational) {
    if (this.stopped || this.busy || this.decidedTimestamp !== this.observedTimestamp ||
        timestamp !== this.observedTimestamp + this.tickMilliseconds)
      throw new Error("Finish requires the next terminal tick after a completed decision");
    this.bot.observe(price);
    try { return this.bot.close(); }
    finally { this.stopped = true; await this.forecaster.close(); }
  }
}
