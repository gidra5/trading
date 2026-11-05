import { advance, terminalClose, type OracleState } from "./oracle-execution.js";
import { observableTree, solveCrra, type Decision, type ForecastTree, type SolverConfig } from "./oracle-crra-policy.js";
import { eq, type Rational } from "./oracle-rational.js";

/** Full-state Oracle bot. The exchange adapter must implement this execution model. */
export class OracleCrraBot {
  private current: OracleState;
  private phase: "decision" | "advance" | "closed" = "decision";
  constructor(initial: OracleState, readonly config: SolverConfig) { this.current = initial; }
  get state(): OracleState { return this.current; }

  decide(observedPrice: Rational, forecast: ForecastTree, certificate = false): Decision {
    if (this.phase !== "decision") throw new Error("Bot requires exactly one decision per observed tick");
    if (!eq(observedPrice, this.current.price)) throw new Error("Forecast observation does not match the account price");
    const tree = observableTree(forecast, this.config.rules.priceStep);
    const decision = solveCrra(this.current, tree, this.config, certificate);
    // No confidence multiplier, grid conversion, target rounding, or second execution.
    this.current = decision.state;
    this.phase = "advance";
    return decision;
  }

  observe(nextPrice: Rational): void {
    if (this.phase !== "advance") throw new Error("Observe requires an executed decision");
    this.current = advance(this.current, nextPrice, this.config.funding(this.current.time), this.config.rules);
    this.phase = "decision";
  }

  close() {
    if (this.phase !== "decision") throw new Error("Close requires a terminal arrival before its decision");
    const result = terminalClose(this.current, this.config.rules);
    this.current = result.state;
    this.phase = "closed";
    return result;
  }
}

export * from "./oracle-crra-policy.js";
export * from "./oracle-execution.js";
export type { Rational, RationalInput } from "./oracle-rational.js";
export * as rational from "./oracle-rational.js";
