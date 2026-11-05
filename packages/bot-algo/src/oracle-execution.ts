import {
  ONE, ZERO, abs, add, div, eq, gt, gte, isZero, lt, lte, min, mul, neg, sub,
  type Rational,
} from "./oracle-rational.js";

/** Oracle.md's received-amount-fee, signed-asset-funding account. */
export interface OracleState {
  readonly quote: Rational;
  readonly asset: Rational;
  readonly price: Rational;
  readonly previousPrice: Rational;
  readonly orders: readonly OracleOrder[];
  readonly time: number;
}
export interface OracleOrder {
  readonly id: string;
  readonly side: "buy" | "sell";
  readonly price: Rational;
  /** Gross quote spent for buys; gross asset value sold for sells. */
  readonly notional: Rational;
  readonly placedAt: number;
}
export interface OracleRules {
  readonly makerFee: Rational;
  readonly takerFee: Rational;
  readonly reserveLeverage: Rational;
  readonly entryLower: Rational;
  readonly entryUpper: Rational;
  readonly maintenanceLower: Rational;
  readonly maintenanceUpper: Rational;
  readonly quoteMinimum: Rational;
  readonly quoteMaximum: Rational;
  readonly priceStep: Rational;
  readonly maxOrders: number;
}
export interface OracleAction {
  /** Actual signed quote movement: negative buys, positive sells. */
  readonly marketQuote: Rational;
  readonly cancelIds: readonly string[];
  readonly place: readonly OracleOrder[];
}
export interface OracleError { readonly code: string; readonly message: string }
export type Result<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: OracleError };
export interface PreparedArrival {
  readonly state: OracleState;
  readonly consumedQuote: Rational;
  readonly fills: readonly OracleOrder[];
}
export interface OracleTerminal {
  readonly state: OracleState;
  /** Marked residual equity after the explicitly supplied one-decision close. */
  readonly wealth: Rational;
  readonly marketQuote: Rational;
}

const success = <T>(value: T): Result<T> => ({ ok: true, value });
const failure = (code: string, message: string): Result<never> =>
  ({ ok: false, error: { code, message } });

export function equity(state: OracleState): Rational {
  return add(state.quote, mul(state.asset, state.price));
}
export function exposure(state: OracleState): Rational {
  const wealth = equity(state);
  return isZero(wealth) ? ZERO : div(mul(state.asset, state.price), wealth);
}
export function effectiveAsset(state: OracleState, rules: OracleRules): Rational {
  const value = mul(state.asset, state.price);
  const received = sub(ONE, rules.takerFee);
  return gte(value, ZERO) ? mul(received, value) : div(value, received);
}
export function effectiveEquity(state: OracleState, rules: OracleRules): Rational {
  return add(state.quote, effectiveAsset(state, rules));
}
export function survives(state: OracleState, rules: OracleRules): boolean {
  const effective = effectiveEquity(state, rules);
  if (!gt(equity(state), ZERO) || !gt(effective, ZERO)) return false;
  const leverage = div(effectiveAsset(state, rules), effective);
  return lte(rules.maintenanceLower, leverage) && lte(leverage, rules.maintenanceUpper);
}
export function reserved(state: OracleState, rules: OracleRules): Rational {
  return state.orders.reduce((total, order) =>
    add(total, div(order.notional, rules.reserveLeverage)), ZERO);
}
export function validQuoteSize(amount: Rational, rules: OracleRules): boolean {
  return isZero(amount) ||
    (lte(rules.quoteMinimum, abs(amount)) && lte(abs(amount), rules.quoteMaximum));
}
export function priceAllowed(price: Rational, rules: OracleRules): boolean {
  return gt(price, ZERO) && gt(rules.priceStep, ZERO) && div(price, rules.priceStep).d === 1n;
}

/** The exact scalar map in OracleOrderLimits.executeQuote; no clamping. */
export function executeQuote(state: OracleState, amount: Rational, fee: Rational): OracleState {
  if (isZero(amount)) return state;
  const received = sub(ONE, fee);
  const quantity = lte(amount, ZERO)
    ? div(mul(received, amount), state.price)
    : div(amount, mul(received, state.price));
  return { ...state, quote: add(state.quote, amount), asset: sub(state.asset, quantity) };
}
export function quoteFlow(order: OracleOrder, fee: Rational): Rational {
  return order.side === "buy" ? neg(order.notional) : mul(sub(ONE, fee), order.notional);
}
export function crossed(state: OracleState, order: OracleOrder): boolean {
  if (order.placedAt >= state.time) return false;
  return order.side === "buy"
    ? lt(state.price, state.previousPrice) && lte(state.price, order.price) && lte(order.price, state.previousPrice)
    : lt(state.previousPrice, state.price) && lte(state.previousPrice, order.price) && lte(order.price, state.price);
}

/** Validate the concrete specialization of LimitOrderAdmissibility.Rules. */
export function validateRules(rules: OracleRules): Result<OracleRules> {
  const zeroFeeDiagnostic = isZero(rules.makerFee) && isZero(rules.takerFee);
  if (!gte(rules.makerFee, ZERO) || (!zeroFeeDiagnostic && !lt(rules.makerFee, rules.takerFee)) || !lt(rules.takerFee, ONE))
    return failure("fees", "Require 0 <= makerFee < takerFee < 1, or both zero for the fee diagnostic.");
  if (!gt(rules.reserveLeverage, ZERO) || !gt(rules.priceStep, ZERO))
    return failure("positive-parameters", "Reserve leverage and price step must be positive.");
  if (!gte(rules.quoteMinimum, ZERO) || !lte(rules.quoteMinimum, rules.quoteMaximum))
    return failure("quote-limits", "Require 0 <= quoteMinimum <= quoteMaximum.");
  if (!lte(rules.entryLower, ZERO) || !gte(rules.entryUpper, ZERO) ||
      !lte(rules.maintenanceLower, ZERO) || !gte(rules.maintenanceUpper, ONE))
    return failure("exposure-limits", "Entry bounds must include cash; maintenance bounds must include [0, 1].");
  if (!Number.isSafeInteger(rules.maxOrders) || rules.maxOrders < 0)
    return failure("order-slots", "maxOrders must be a nonnegative safe integer.");
  return success(rules);
}
function validateOrder(order: OracleOrder, rules: OracleRules): Result<OracleOrder> {
  if (typeof order.id !== "string" || (order.side !== "buy" && order.side !== "sell"))
    return failure("order-identity", "Orders require a string ID and buy or sell side.");
  if (!priceAllowed(order.price, rules) || !gt(order.notional, ZERO))
    return failure("order-domain", "Order prices must be positive grid points and notionals positive.");
  if (!Number.isSafeInteger(order.placedAt) || order.placedAt < 0)
    return failure("order-time", "Placement time must be a nonnegative safe integer.");
  return success(order);
}
function validBook(book: readonly OracleOrder[], rules: OracleRules): Result<readonly OracleOrder[]> {
  if (book.length > rules.maxOrders) return failure("order-slots", "Pending book exceeds maxOrders.");
  const ids = new Set<string>();
  for (const order of book) {
    const check = validateOrder(order, rules);
    if (!check.ok) return check;
    if (ids.has(order.id)) return failure("duplicate-order", "Active order IDs must be distinct.");
    ids.add(order.id);
  }
  return success(book);
}
function validState(state: OracleState, rules: OracleRules): Result<OracleState> {
  const ruleCheck = validateRules(rules);
  if (!ruleCheck.ok) return ruleCheck;
  if (!Number.isSafeInteger(state.time) || state.time < 0)
    return failure("time", "Decision time must be a nonnegative safe integer.");
  if (!priceAllowed(state.price, rules) || !gt(state.previousPrice, ZERO))
    return failure("price", "Current price must be a positive grid point and previous price positive.");
  const bookCheck = validBook(state.orders, rules);
  return bookCheck.ok ? success(state) : bookCheck;
}

/** Arrival happens before cancellations. Invalid committed full fills are rejected, never repaired. */
export function prepareArrival(state: OracleState, rules: OracleRules): Result<PreparedArrival> {
  const check = validState(state, rules);
  if (!check.ok) return check;
  const fills: OracleOrder[] = [];
  const pending: OracleOrder[] = [];
  let account = state;
  let consumedQuote = ZERO;
  for (const order of state.orders) {
    if (!crossed(state, order)) { pending.push(order); continue; }
    const amount = quoteFlow(order, rules.makerFee);
    if (!validQuoteSize(amount, rules))
      return failure("fill-size", "A committed full fill violates its quote-size bounds.");
    consumedQuote = add(consumedQuote, abs(amount));
    if (!lte(consumedQuote, rules.quoteMaximum))
      return failure("aggregate-quote-cap", "Committed fills exceed this decision's aggregate quote cap.");
    // Execution uses the order price; marking remains at the observed arrival price.
    account = { ...executeQuote({ ...account, price: order.price }, amount, rules.makerFee), price: state.price };
    fills.push(order);
  }
  return success({ state: fills.length ? { ...account, orders: pending } : state, consumedQuote, fills });
}

/** Concrete accountAllowed: positive raw equity, plus absorbing ruin. */
export function accountAllowed(state: OracleState, rules: OracleRules): boolean {
  if (isZero(state.quote) && isZero(state.asset)) return true;
  return gt(equity(state), ZERO);
}

/** Execute one exact market/cancel/place decision on an already prepared arrival. */
export function executeAction(prepared: PreparedArrival, action: OracleAction, rules: OracleRules): Result<OracleState> {
  const state = prepared.state;
  const check = validState(state, rules);
  if (!check.ok) return check;
  if (!gte(prepared.consumedQuote, ZERO) || !validQuoteSize(action.marketQuote, rules) ||
      !lte(add(prepared.consumedQuote, abs(action.marketQuote)), rules.quoteMaximum))
    return failure("aggregate-quote-cap", "Market order and arrival fills must obey the shared decision budget and order minimum.");
  const cancelIds = new Set(action.cancelIds);
  const remaining = state.orders.filter((order) => !cancelIds.has(order.id));
  const canceled = remaining.length === state.orders.length ? state : { ...state, orders: remaining };
  if (!lte(reserved(canceled, rules), equity(state)))
    return failure("arrival-reservation", "Remaining reservations exceed arrival equity.");
  for (const order of action.place) {
    const orderCheck = validateOrder(order, rules);
    if (!orderCheck.ok) return orderCheck;
    if (order.placedAt !== state.time || !(order.side === "buy" ? lt(order.price, state.price) : gt(order.price, state.price)))
      return failure("passive-placement", "New orders must be strictly passive and timestamped at this decision.");
    if (!validQuoteSize(quoteFlow(order, rules.makerFee), rules))
      return failure("placement-size", "New order's quote flow must obey the physical order bounds.");
  }
  const traded = executeQuote(canceled, action.marketQuote, rules.takerFee);
  const next = action.place.length ? { ...traded, orders: [...traded.orders, ...action.place] } : traded;
  const bookCheck = validBook(next.orders, rules);
  if (!bookCheck.ok) return bookCheck;
  const reserve = reserved(next, rules);
  if (!lte(reserve, equity(next)))
    return failure("reservation", "Pending reservations exceed post-rebalance equity.");
  if (next.orders.length && !lte(add(div(abs(mul(next.asset, next.price)), rules.reserveLeverage), reserve), equity(next)))
    return failure("shared-collateral", "Position margin and pending reservations exceed shared account equity.");
  if (!accountAllowed(next, rules))
    return failure("raw-equity", "Post-rebalance account must have positive raw equity or be absorbing ruin.");
  // Concrete `additional` predicate: physical rebalances respect the entry band.
  // Coasting can retain exposure drift outside that band until maintenance fails.
  if ((!isZero(action.marketQuote) || prepared.fills.length > 0) &&
      (!lte(rules.entryLower, exposure(next)) || !lte(exposure(next), rules.entryUpper)))
    return failure("entry-exposure", "Post-trade exposure exceeds the entry band.");
  return success(next);
}

/** Funding/marking and liquidation occur before the following arrival fills. */
export function advance(state: OracleState, nextPrice: Rational, fundingFactor: Rational, rules: OracleRules): OracleState {
  if (!priceAllowed(nextPrice, rules) || !gt(fundingFactor, ZERO))
    throw new Error("Oracle advance requires a positive grid price and positive funding factor.");
  if (!Number.isSafeInteger(state.time + 1)) throw new Error("Oracle decision time overflow.");
  const moved: OracleState = {
    ...state, asset: mul(state.asset, fundingFactor), previousPrice: state.price,
    price: nextPrice, time: state.time + 1,
  };
  return survives(moved, rules) ? moved : { ...moved, quote: ZERO, asset: ZERO, orders: [] };
}

/** Exact fee-aware request; callers choose a valid action, without hidden target clamping. */
export function requestForTarget(state: OracleState, target: Rational, rules: OracleRules): Rational {
  const wealth = equity(state);
  if (!gt(wealth, ZERO)) {
    if (isZero(state.quote) && isZero(state.asset) && isZero(target)) return ZERO;
    throw new Error("Target requests require positive equity, except absorbing cash at ruin.");
  }
  const current = exposure(state);
  if (eq(current, target)) return ZERO;
  const received = sub(ONE, rules.takerFee);
  const denominator = lt(current, target)
    ? add(received, mul(rules.takerFee, target))
    : sub(ONE, mul(rules.takerFee, target));
  if (!gt(denominator, ZERO)) throw new Error("Target lies outside the positive rebalance-denominator domain.");
  return lt(current, target)
    ? neg(div(mul(wealth, sub(target, current)), denominator))
    : div(mul(mul(received, wealth), sub(current, target)), denominator);
}

/**
 * One physical closing decision: process arrival, cancel the remaining orders,
 * and close within the remaining aggregate budget. Retain any residual exactly.
 * This operation alone does not imply a feasible completed policy.
 */
export function closeWithinCapacityChecked(state: OracleState, rules: OracleRules): Result<OracleTerminal> {
  const prepared = prepareArrival(state, rules);
  if (!prepared.ok) return prepared;
  const arrived = prepared.value.state;
  const available = abs(effectiveAsset(arrived, rules));
  const remainingBudget = sub(rules.quoteMaximum, prepared.value.consumedQuote);
  const maximum = min(available, remainingBudget);
  const amount = lt(maximum, rules.quoteMinimum) ? ZERO : maximum;
  const marketQuote = lte(effectiveAsset(arrived, rules), ZERO) ? neg(amount) : amount;
  const canceled = arrived.orders.length ? { ...arrived, orders: [] } : arrived;
  const closed = executeQuote(canceled, marketQuote, rules.takerFee);
  return success({ state: closed, wealth: equity(closed), marketQuote });
}

/** A completed policy must reach exactly zero inventory and an empty order book.
 * Dust or insufficient final capacity makes the continuation infeasible; earlier
 * rebalances must arrange a legal full close. No extra closing tick is granted.
 */
export function terminalCloseChecked(state: OracleState, rules: OracleRules): Result<OracleTerminal> {
  const result = closeWithinCapacityChecked(state, rules);
  if (!result.ok) return result;
  if (!isZero(result.value.state.asset))
    return failure("terminal-exposure", "The horizon must end at exactly zero holdings; close earlier within the order caps.");
  if (lt(result.value.wealth, ZERO))
    return failure("terminal-equity", "Completed terminal wealth must be nonnegative.");
  return result;
}

export function terminalClose(state: OracleState, rules: OracleRules): OracleTerminal {
  const result = terminalCloseChecked(state, rules);
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
}
