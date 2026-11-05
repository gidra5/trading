import { advance, equity, executeAction, prepareArrival, terminalCloseChecked, validateRules, type OracleAction, type OracleRules, type OracleState } from './oracle-execution.js';
import { expectedUtility, type Utility, type PriceTree, type SolverConfig } from './oracle-crra-policy.js';
import { ZERO, ONE, rat, abs, add, sub, mul, div, neg, gt, lt, eq, sum, serializeRat, type Rational } from './oracle-rational.js';

interface Line { readonly slope: Rational; readonly intercept: Rational }
export interface ShadowFrame {
  readonly price: Rational;
  readonly previousPrice: Rational;
  /** Product of actual signed-asset funding multipliers since the initial account. */
  readonly cumulativeFunding: Rational;
}
export interface ShadowPathResult { readonly upper: Rational; readonly referencePrice: Rational }

const max = (a: Rational, b: Rational) => gt(a,b) ? a : b;
const evaluate = (line: Line, z: Rational) => add(mul(line.slope,z),line.intercept);

/** Trade gain per absolute quote unit when valuing asset at z/cumulativeFunding.
 * A downward move can only fill buys, at or above the new mark. An upward
 * move can only fill sells, at or below the new mark. At z>=0, moving those
 * fills to the new mark only increases their shadow gain. Other maker-side
 * trades are NOT permitted by this upper relaxation.
 * Zero and both market directions are always included. Each frame has ONE cap.
 */
function frameLines(frame: ShadowFrame, rules: OracleRules): readonly Line[] {
  if (!gt(frame.price,ZERO) || !gt(frame.previousPrice,ZERO) || !gt(frame.cumulativeFunding,ZERO))
    throw new Error('Positive prices and cumulative funding required');
  const p = mul(frame.price,frame.cumulativeFunding);
  const buy = (fee: Rational): Line => ({ slope: div(sub(ONE,fee),p), intercept: neg(ONE) });
  const sell = (fee: Rational): Line => ({ slope: neg(div(ONE,mul(sub(ONE,fee),p))), intercept: ONE });
  return [{ slope: ZERO, intercept: ZERO }, buy(rules.takerFee), sell(rules.takerFee),
    ...(lt(frame.price,frame.previousPrice) ? [buy(rules.makerFee)] : []),
    ...(gt(frame.price,frame.previousPrice) ? [sell(rules.makerFee)] : [])];
}

/** Pathwise upper bound for every valid continuous Oracle execution ending flat.
 * Controls, shared margins and order lifecycles are relaxed, never restricted.
 * Nonnegative shadow capital survives zero-account liquidation. The last frame
 * is the actual terminal arrival/close, sharing its ordinary aggregate cap.
 * Choosing z with hindsight is legal for an UPPER bound, not for a trading policy.
 */
export function shadowPathUpper(initial: Pick<OracleState,'quote'|'asset'>,
  frames: readonly ShadowFrame[], rules: OracleRules): ShadowPathResult {
  if (lt(rules.makerFee,ZERO) || lt(rules.takerFee,rules.makerFee) || !lt(rules.takerFee,ONE) || lt(rules.quoteMaximum,ZERO))
    throw new Error('Invalid shadow-bound fees or cap');
  const groups: readonly (readonly Line[])[] = [
    [{ slope: ZERO, intercept: ZERO }, { slope: initial.asset, intercept: initial.quote }],
    ...frames.map(frame => frameLines(frame,rules)),
  ];
  // A convex piecewise-linear function attains a minimum at zero or a crossing
  // of two lines in a group. Every tested z gives a valid upper bound even
  // without relying on the completeness of this minimization procedure.
  const candidates = new Map<string,Rational>([['0',ZERO]]);
  for (const lines of groups) for (let i=0;i<lines.length;i++) for (let j=i+1;j<lines.length;j++) {
    const difference = sub(lines[i]!.slope,lines[j]!.slope);
    if (eq(difference,ZERO)) continue;
    const z = div(sub(lines[j]!.intercept,lines[i]!.intercept),difference);
    if (!lt(z,ZERO)) candidates.set(serializeRat(z),z);
  }
  let best: ShadowPathResult | undefined;
  for (const z of candidates.values()) {
    const terms = groups.map(lines => lines.reduce((value,line) => max(value,evaluate(line,z)),ZERO));
    const upper = add(terms[0]!,mul(rules.quoteMaximum,sum(terms.slice(1))));
    if (!best || lt(upper,best.upper)) best = { upper, referencePrice: z };
  }
  return best!;
}

export interface ShadowTreeResult {
  readonly upper: Rational;
  readonly cashUpper: Rational;
  readonly paths: readonly { readonly probability: Rational; readonly frames: readonly ShadowFrame[]; readonly bound: ShadowPathResult }[];
}

/** Exact expectation of pathwise upper bounds for the supplied finite law.
 * This covers continuous controls at ALL modeled decision times, including
 * arbitrary valid limit orders. It does not certify omitted times or NSF tails.
 */
export function shadowTreeUpper(state: OracleState, tree: PriceTree, config: SolverConfig,
  firstAction?: OracleAction): ShadowTreeResult {
  if (config.riskAversion !== 0 || (firstAction && !tree.children.length)) throw new Error('Require gamma=0 and a future transition after a fixed action');
  let initial = state;
  if (firstAction) {
    const arrival = prepareArrival(state,config.rules);
    if (!arrival.ok) throw new Error(arrival.error.message);
    const executed = executeAction(arrival.value,firstAction,config.rules);
    if (!executed.ok) throw new Error(executed.error.message);
    initial = executed.value;
  }
  // When no first action is fixed, include the current arrival/rebalance or
  // terminal close in the shadow budget. No extra decision is granted.
  const initialFrames: readonly ShadowFrame[] = firstAction ? [] : [{price:state.price,previousPrice:state.previousPrice,cumulativeFunding:ONE}];
  const paths: ShadowTreeResult['paths'][number][] = [];
  const walk = (node: PriceTree, price: Rational, funding: Rational, weight: Rational, frames: readonly ShadowFrame[], depth: number) => {
    if (!node.children.length) {
      paths.push({ probability: weight, frames, bound: shadowPathUpper(initial,frames,config.rules) });
      return;
    }
    if (!eq(sum(node.children.map(b=>b.probability)),ONE) || node.children.some(b=>!gt(b.probability,ZERO)))
      throw new Error('Positive normalized conditional probabilities required');
    const factor = config.funding(state.time+depth);
    if (!gt(factor,ZERO)) throw new Error('Positive funding required');
    const cumulative = mul(funding,factor);
    for (const branch of node.children) walk(branch,branch.price,cumulative,mul(weight,branch.probability),
      [...frames,{ price: branch.price, previousPrice: price, cumulativeFunding: cumulative }],depth+1);
  };
  walk(tree,state.price,ONE,ONE,initialFrames,0);
  const upper = sum(paths.map(p=>mul(p.probability,p.bound.upper)));
  const cashUpper = sum(paths.map(p=>mul(p.probability,add(max(initial.quote,ZERO),mul(rat(p.frames.length),config.rules.quoteMaximum)))));
  if (gt(upper,cashUpper)) throw new Error('Shadow optimization must include the cash bound z=0');
  return { upper, cashUpper, paths };
}

export interface MartingaleShadowResult {
  readonly upper: Rational;
  readonly rawLeverageBound: Rational;
  readonly maximumScaledEdgeMove: Rational;
}

/** A NONANTICIPATIVE shadow-price bound. Terminal reference equals terminal
 * market price; references are backed up as z_parent=k*E[z_child]. Trading
 * inventory is chosen before the child is observed, so funding/marking makes
 * no expected shadow gain. The checked edge band keeps every possible moved
 * shadow balance nonnegative, so liquidation cannot add value to it.
 *
 * With static Oracle rules, surviving inventory has raw leverage <= max(1,
 * absolute maintenance bounds). Every physical trade targets the entry band;
 * a no-trade rebalance preserves the surviving account. Thus L below covers
 * every feasible post-rebalance state. An initial account outside it, or an
 * uncovered edge, returns null and MUST NOT be treated as a zero upper bound.
 */
export function martingaleShadowUpper(state: OracleState, tree: PriceTree, config: SolverConfig,
  firstAction?: OracleAction): MartingaleShadowResult | null {
  if(config.riskAversion!==0)throw new Error('Martingale shadow requires gamma=0');
  const checked=validateRules(config.rules); if(!checked.ok)throw new Error(checked.error.message);
  let initial=state;
  if(firstAction) {
    if(!tree.children.length)throw new Error('Fixed action requires a future transition');
    const arrival=prepareArrival(state,config.rules); if(!arrival.ok)return null;
    const executed=executeAction(arrival.value,firstAction,config.rules); if(!executed.ok)return null;
    initial=executed.value;
  }
  const L=[ONE,abs(config.rules.entryLower),abs(config.rules.entryUpper),abs(config.rules.maintenanceLower),abs(config.rules.maintenanceUpper)].reduce(max,ZERO);
  if(lt(equity(initial),ZERO)||gt(abs(mul(initial.asset,initial.price)),mul(L,equity(initial))))return null;
  let maximumScaledEdgeMove=ZERO;
  const build=(node:PriceTree,price:Rational,previousPrice:Rational,depth:number):{reference:Rational;cost:Rational}|null=>{
    let reference=price,continuationCost=ZERO;
    if(node.children.length) {
      if(!eq(sum(node.children.map(b=>b.probability)),ONE)||node.children.some(b=>!gt(b.probability,ZERO)))throw new Error('Positive normalized probabilities required');
      const k=config.funding(state.time+depth); if(!gt(k,ZERO))throw new Error('Positive funding required');
      const children=[];
      for(const b of node.children) {
        const child=build(b,b.price,price,depth+1); if(!child)return null;
        const scaled=mul(L,abs(sub(div(mul(k,child.reference),price),ONE)));
        maximumScaledEdgeMove=max(maximumScaledEdgeMove,scaled);
        if(gt(scaled,ONE))return null;
        children.push({weight:b.probability,...child});
      }
      reference=mul(k,sum(children.map(c=>mul(c.weight,c.reference))));
      continuationCost=sum(children.map(c=>mul(c.weight,c.cost)));
    }
    const coefficient=frameLines({price,previousPrice,cumulativeFunding:ONE},config.rules)
      .reduce((value,line)=>max(value,evaluate(line,reference)),ZERO);
    const cost=add(continuationCost,(depth===0&&firstAction)?ZERO:mul(config.rules.quoteMaximum,coefficient));
    return {reference,cost};
  };
  const root=build(tree,state.price,state.previousPrice,0); if(!root)return null;
  const upper=add(max(add(initial.quote,mul(initial.asset,root.reference)),ZERO),root.cost);
  return {upper,rawLeverageBound:L,maximumScaledEdgeMove};
}

export function continuousShadowUpper(state:OracleState,tree:PriceTree,config:SolverConfig,firstAction?:OracleAction):Rational {
  const path=shadowTreeUpper(state,tree,config,firstAction).upper;
  const martingale=martingaleShadowUpper(state,tree,config,firstAction);
  return martingale&&lt(martingale.upper,path)?martingale.upper:path;
}

/** Refine the pathwise bound using actual nonanticipative Bellman decisions.
 * Only the SUPPLIED FINITE MENU is covered by this refinement. The direct
 * shadowTreeUpper continues to cover continuous controls at the modeled times.
 * Each refinement intersects the previous pathwise bound with the Bellman
 * backup, so refinement cannot increase the bound. The actual terminal close
 * is used at true leaves; empty/infeasible menus remain bottom (null).
 * depth counts evaluated account decisions, independently of the cutoff H
 * used for the relative-contribution denominator.
 */
export function refinedShadowUpper(state: OracleState, tree: PriceTree, config: SolverConfig,
  depth: number, firstAction?: OracleAction): Utility {
  if (!Number.isSafeInteger(depth) || depth<0 || config.riskAversion!==0) throw new Error('Require nonnegative integer depth and gamma=0');
  if (!tree.children.length) {
    const terminal=terminalCloseChecked(state,config.rules);
    return terminal.ok ? terminal.value.wealth : null;
  }
  const direct=continuousShadowUpper(state,tree,config,firstAction);
  if (!depth) return direct;
  const prepared=prepareArrival(state,config.rules);
  if (!prepared.ok) return null;
  const actions=firstAction?[firstAction]:config.menu(prepared.value,tree,config.rules).map(x=>x.action);
  let best: Utility=null;
  for (const action of actions) {
    const executed=executeAction(prepared.value,action,config.rules);
    if (!executed.ok) continue;
    const outcomes=tree.children.map(branch=>({probability:branch.probability,
      utility:refinedShadowUpper(advance(executed.value,branch.price,config.funding(state.time),config.rules),branch,config,depth-1)}));
    const value=expectedUtility(outcomes);
    if (value!==null && (best===null || gt(value,best))) best=value;
  }
  return best===null ? null : (lt(best,direct)?best:direct);
}
