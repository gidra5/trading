import { advance, executeAction, prepareArrival, terminalCloseChecked, type OracleAction, type OracleState } from './oracle-execution.js';
import { NoFeasiblePolicy, type NamedAction, type PriceTree, type SolverConfig } from './oracle-crra-policy.js';
import { continuousShadowUpper, shadowTreeUpper } from './oracle-shadow-tail.js';
import { ZERO, ONE, rat, fraction, add, sub, mul, div, gt, lt, eq, sum, type Rational } from './oracle-rational.js';

/** Natural logarithms are enclosed, never passed off as exact rationals.
 * Utilities are log(terminal wealth / wealthUnit), an additive normalization
 * which does not change the optimal policy. Null is negative infinity.
 */
export interface LogInterval { readonly lower: Rational; readonly upper: Rational }
export type LogUtility = LogInterval | null;
export interface LogSolverConfig extends Omit<SolverConfig, 'riskAversion'> {
  readonly riskAversion: 1;
  readonly wealthUnit: Rational;
}
const SCALE = 1n << 64n;
const TERMS = 32;
const ceilPositive = (n: bigint, d: bigint) => (n+d-1n)/d;

/** For 1 <= n/d <= 2, y=(x-1)/(x+1) lies in [0,1/3].
 * log(x)=2 sum_(j>=0) y^(2j+1)/(2j+1). All fixed-point operations
 * round outwards. The omitted positive tail is <= (9/4)*3^(-2N-1).
 */
function mantissaLog(n: bigint, d: bigint): readonly [bigint,bigint] {
  if(n<d || n>2n*d)throw new Error('Log mantissa outside [1,2]');
  if(n===d)return [0n,0n];
  const yLow=SCALE*(n-d)/(n+d),yHigh=ceilPositive(SCALE*(n-d),n+d);
  const y2Low=yLow*yLow/SCALE,y2High=ceilPositive(yHigh*yHigh,SCALE);
  let powerLow=yLow,powerHigh=yHigh,low=0n,high=0n;
  for(let j=0;j<TERMS;j++) {
    const denominator=BigInt(2*j+1);
    low+=powerLow/denominator;
    high+=ceilPositive(powerHigh,denominator);
    powerLow=powerLow*y2Low/SCALE;
    powerHigh=ceilPositive(powerHigh*y2High,SCALE);
  }
  return [2n*low,2n*high+ceilPositive(9n*SCALE,4n*3n**BigInt(2*TERMS+1))];
}
const LOG_TWO=mantissaLog(2n,1n);

export function logInterval(x: Rational): LogInterval {
  if(!gt(x,ZERO))throw new Error('Log requires positive wealth');
  let k=x.n.toString(2).length-x.d.toString(2).length;
  let n=x.n,d=x.d;
  if(k>=0)d<<=BigInt(k);else n<<=BigInt(-k);
  if(n<d) {n*=2n;k--;}
  const [lo,hi]=mantissaLog(n,d);
  const shift=BigInt(k);
  return {lower:fraction(lo+shift*LOG_TWO[k>=0?0:1],SCALE),
    upper:fraction(hi+shift*LOG_TWO[k>=0?1:0],SCALE)};
}

export function expectedLog(branches:readonly {readonly probability:Rational;readonly value:LogUtility}[]):LogUtility {
  if(branches.some(b=>lt(b.probability,ZERO))||!eq(sum(branches.map(b=>b.probability)),ONE))
    throw new Error('Expected log requires normalized nonnegative probabilities');
  let lower=ZERO,upper=ZERO;
  for(const b of branches) {
    if(eq(b.probability,ZERO))continue;
    if(b.value===null)return null;
    lower=add(lower,mul(b.probability,b.value.lower));
    upper=add(upper,mul(b.probability,b.value.upper));
  }
  return {lower,upper};
}
const maximum=(a:Rational,b:Rational)=>gt(a,b)?a:b;
const minimum=(a:Rational,b:Rational)=>lt(a,b)?a:b;

export interface LogDecision extends NamedAction {
  /** Encloses the supremum over the declared finite menu. The lower endpoint
   * is attained or exceeded by this recursively selected feasible policy.
   * Overlapping action intervals are retained in the upper endpoint.
   */
  readonly value:LogUtility;
  readonly state:OracleState;
  readonly expectedWealth:Rational;
  readonly feasibleCandidates:number;
}

/** Finite nonanticipative Bellman induction with certified log intervals.
 * It certifies a policy's optimality gap; an overlapping interval is not
 * silently claimed to identify an exact maximizing action.
 */
export function solveLogCrra(state:OracleState,tree:PriceTree,config:LogSolverConfig):LogDecision {
  if(config.riskAversion!==1||!gt(config.wealthUnit,ZERO))throw new Error('Require gamma=1 and a positive wealth unit');
  if(!tree.children.length)throw new Error('A decision requires a future tick');
  if(tree.children.some(b=>!gt(b.probability,ZERO))||!eq(sum(tree.children.map(b=>b.probability)),ONE))
    throw new Error('Use positive normalized conditional probabilities');
  if(new Set(tree.children.map(b=>JSON.stringify([b.price.n.toString(),b.price.d.toString(),b.observationKey??'']))).size!==tree.children.length)
    throw new Error('Indistinguishable branches must be merged with observableTree');
  const prepared=prepareArrival(state,config.rules);
  if(!prepared.ok)throw new NoFeasiblePolicy(prepared.error.message);
  const ids=new Set<string>();
  let best:LogDecision|undefined,upper:Rational|null=null,feasibleCandidates=0;
  for(const named of config.menu(prepared.value,tree,config.rules)) {
    if(ids.has(named.id))throw new Error('Duplicate action ID');ids.add(named.id);
    const executed=executeAction(prepared.value,named.action,config.rules);
    if(!executed.ok)continue;
    const outcomes:{probability:Rational;value:LogUtility}[]=[];
    let expectedWealth=ZERO,feasible=true;
    for(const branch of tree.children) {
      const moved=advance(executed.value,branch.price,config.funding(state.time),config.rules);
      let value:LogUtility,wealth:Rational;
      if(!branch.children.length) {
        const terminal=terminalCloseChecked(moved,config.rules);
        if(!terminal.ok){feasible=false;break;}
        wealth=terminal.value.wealth;
        value=gt(wealth,ZERO)?logInterval(div(wealth,config.wealthUnit)):null;
      } else {
        try {const child=solveLogCrra(moved,branch,config);value=child.value;wealth=child.expectedWealth;}
        catch(error){if(!(error instanceof NoFeasiblePolicy))throw error;feasible=false;break;}
      }
      outcomes.push({probability:branch.probability,value});
      expectedWealth=add(expectedWealth,mul(branch.probability,wealth));
    }
    if(!feasible)continue;
    feasibleCandidates++;
    const value=expectedLog(outcomes);
    if(value!==null)upper=upper===null?value.upper:maximum(upper,value.upper);
    if(!best || (value!==null && (best.value===null||gt(value.lower,best.value.lower))))
      best={...named,state:executed.value,value,expectedWealth,feasibleCandidates:0};
  }
  if(!best)throw new NoFeasiblePolicy('No valid flat-ending policy in the declared menu');
  return {...best,feasibleCandidates,value:best.value===null?null:{lower:best.value.lower,upper:upper!}};
}

/** Upper expected log for EVERY continuous control on the supplied finite law.
 * Monotonicity gives E log W <= E log(pathwise wealth upper).
 * Jensen separately gives E log W <= log(E W upper). Take their minimum.
 * The gamma-zero config below computes a wealth envelope, not a log policy.
 */
export function logShadowUpper(state:OracleState,tree:PriceTree,config:LogSolverConfig,firstAction?:OracleAction):Rational|null {
  const wealthConfig:SolverConfig={...config,riskAversion:0};
  const paths=shadowTreeUpper(state,tree,wealthConfig,firstAction).paths;
  const pathLog=expectedLog(paths.map(p=>({probability:p.probability,
    value:gt(p.bound.upper,ZERO)?logInterval(div(p.bound.upper,config.wealthUnit)):null})));
  const wealthUpper=continuousShadowUpper(state,tree,wealthConfig,firstAction);
  if(pathLog===null||!gt(wealthUpper,ZERO))return null;
  return minimum(pathLog.upper,logInterval(div(wealthUpper,config.wealthUnit)).upper);
}

export function refinedLogShadowUpper(state:OracleState,tree:PriceTree,config:LogSolverConfig,
  depth:number,firstAction?:OracleAction):Rational|null {
  if(!Number.isSafeInteger(depth)||depth<0)throw new Error('Require nonnegative integer refinement depth');
  if(!tree.children.length) {
    const terminal=terminalCloseChecked(state,config.rules);
    return terminal.ok&&gt(terminal.value.wealth,ZERO)?logInterval(div(terminal.value.wealth,config.wealthUnit)).upper:null;
  }
  const direct=logShadowUpper(state,tree,config,firstAction);
  if(!depth||direct===null)return direct;
  const prepared=prepareArrival(state,config.rules);if(!prepared.ok)return null;
  const actions=firstAction?[firstAction]:config.menu(prepared.value,tree,config.rules).map(n=>n.action);
  let best:Rational|null=null;
  for(const action of actions) {
    const executed=executeAction(prepared.value,action,config.rules);if(!executed.ok)continue;
    const children=tree.children.map(b=>{
      const value=refinedLogShadowUpper(advance(executed.value,b.price,config.funding(state.time),config.rules),b,config,depth-1);
      return {probability:b.probability,value:value===null?null:{lower:value,upper:value}};
    });
    const expected=expectedLog(children);
    if(expected!==null)best=best===null?expected.upper:maximum(best,expected.upper);
  }
  return best===null?null:minimum(direct,best);
}

/** No floating exponential is needed for the strict 0.1% certificate.
 * exp(delta)-1 < eps iff delta < log(1+eps). Overlap stays undecided.
 */
export function belowCeTolerance(deltaUpper:Rational,epsilon:Rational):boolean {
  if(!gt(epsilon,ZERO))throw new Error('Require positive tolerance');
  return lt(deltaUpper,logInterval(add(ONE,epsilon)).lower);
}
