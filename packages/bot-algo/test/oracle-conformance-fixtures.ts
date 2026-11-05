import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ONE, ZERO, rat, serializeRat, type Rational } from "../src/oracle-rational.js";
import {
  advance, effectiveEquity, equity, executeAction, executeQuote, prepareArrival, survives,
  type OracleAction, type OracleRules, type OracleState, type Result,
} from "../src/oracle-execution.js";

const baseRules: OracleRules = {
  makerFee: ZERO, takerFee: rat("0.001"), reserveLeverage: rat(5),
  entryLower: rat(-5), entryUpper: rat(5), maintenanceLower: rat(-2), maintenanceUpper: rat(2),
  quoteMinimum: ZERO, quoteMaximum: rat(1000), priceStep: rat("0.01"), maxOrders: 4,
};
const source = (quote: string, asset: string, price = "100"): OracleState => ({
  quote: rat(quote), asset: rat(asset), price: rat(price), previousPrice: rat(price), orders: [], time: 0,
});
const account = (state: OracleState) => ({
  quote: serializeRat(state.quote), asset: serializeRat(state.asset), price: serializeRat(state.price),
});
function must<T>(result: Result<T>): T {
  if (!result.ok) throw new Error(JSON.stringify(result.error));
  return result.value;
}
export interface OracleConformanceFixture {
  readonly kind: "oracle";
  readonly operation: "executeQuote" | "advance" | "step" | "metrics";
  readonly state: ReturnType<typeof account>;
  readonly fee: string;
  readonly lower: string;
  readonly upper: string;
  readonly dq: string;
  readonly factor: string;
  readonly nextPrice: string;
  readonly expected: ReturnType<typeof account> & {
    readonly equity: string;
    readonly effectiveEquity: string;
    readonly survives: boolean;
  };
}

/** Exact outputs consumed by the independent, proved Lean rational kernel. */
export function oracleConformanceFixtures(): readonly OracleConformanceFixture[] {
  const fixtures: OracleConformanceFixture[] = [];
  const append = (
    operation: OracleConformanceFixture["operation"], state: OracleState,
    dq = ZERO, factor = ONE, nextPrice = state.price, config = baseRules,
  ) => {
    let next = state;
    if (operation === "executeQuote" || operation === "step") next = executeQuote(next, dq, config.takerFee);
    if (operation === "advance" || operation === "step") next = advance(next, nextPrice, factor, config);
    fixtures.push({
      kind: "oracle", operation, state: account(state), fee: serializeRat(config.takerFee),
      lower: serializeRat(config.maintenanceLower), upper: serializeRat(config.maintenanceUpper),
      dq: serializeRat(dq), factor: serializeRat(factor), nextPrice: serializeRat(nextPrice),
      expected: {
        ...account(next), equity: serializeRat(equity(next)), effectiveEquity: serializeRat(effectiveEquity(next, config)),
        survives: survives(next, config),
      },
    });
  };
  // Both quote branches, zero, crosses flat, negative balances, and small rational amounts.
  for (const state of [source("100", "0"), source("0", "1"), source("200", "-1"), source("-50", "1"), source("0", "0")]) {
    append("metrics", state);
    for (const dq of ["-100", "-1/7", "0", "1/7", "99.9", "150"]) append("executeQuote", state, rat(dq));
    for (const factor of ["0.9", "1", "1.1"]) {
      for (const price of ["40", "100", "160"]) append("advance", state, ZERO, rat(factor), rat(price));
    }
  }
  for (const fee of ["0.0005", "0.01", "0.1"]) {
    const config = { ...baseRules, takerFee: rat(fee) };
    for (const dq of ["-100", "100"]) for (const price of ["50", "100", "150"])
      append("step", source("100", "0"), rat(dq), rat("0.99"), rat(price), config);
  }
  // Long effective exposure is exactly two, immediately outside, and immediately inside.
  for (const quote of ["-49.95", "-49.951", "-49.949"]) {
    append("metrics", source(quote, "1"));
    append("advance", source(quote, "1"));
  }
  // Short effective exposure is exactly minus two with fee-aware short closeout.
  const shortBoundary = source("150", "-0.999");
  append("metrics", shortBoundary);
  append("advance", shortBoundary);
  // Price quantization and residual/dust positions remain exact in the core map.
  append("executeQuote", source("1/3", "1/7", "0.01"), rat("-1/100000"));
  append("metrics", source("3/4", "1/4", "1"));
  return fixtures;
}

/** Whole-book data for original LimitOrderMachine/Admissibility concrete Lean fixtures. */
export function oracleLimitConformanceTrace() {
  const rules = { ...baseRules, quoteMaximum: rat(200), quoteMinimum: rat(1) };
  const initial = source("1000", "0");
  const first: OracleAction = {
    marketQuote: rat(-100), cancelIds: [], place: [
      { id: "1", side: "buy", price: rat(95), notional: rat(95), placedAt: 0 },
      { id: "2", side: "sell", price: rat(110), notional: rat(110), placedAt: 0 },
    ],
  };
  const afterFirst = must(executeAction(must(prepareArrival(initial, rules)), first, rules));
  const moved = advance(afterFirst, rat(90), ONE, rules);
  const arrival = must(prepareArrival(moved, rules));
  const second: OracleAction = {
    marketQuote: rat("89.91"), cancelIds: ["2"], place: [
      { id: "3", side: "buy", price: rat(85), notional: rat(85), placedAt: 1 },
    ],
  };
  const afterSecond = must(executeAction(arrival, second, rules));
  const final = advance(afterSecond, rat(100), rat("0.9"), rules);
  const serialize = (value: unknown): unknown => {
    if (value && typeof value === "object" && "n" in value && "d" in value)
      return serializeRat(value as Rational);
    if (Array.isArray(value)) return value.map(serialize);
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, serialize(v)]));
    return value;
  };
  return serialize({ rules, initial, first, afterFirst, moved, arrival, second, afterSecond, final });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const output = resolve(process.argv[2] ?? "data/backtests/stochastic-crra/oracle-conformance.jsonl");
  mkdirSync(dirname(output), { recursive: true });
  const records = oracleConformanceFixtures();
  writeFileSync(output, records.map((record) => JSON.stringify(record)).join("\n") + "\n");
  writeFileSync(output.replace(/\.jsonl$/, "-limits.json"), JSON.stringify(oracleLimitConformanceTrace(), null, 2) + "\n");
  process.stdout.write(`Wrote ${records.length} exact Oracle fixtures to ${output}\n`);
}
