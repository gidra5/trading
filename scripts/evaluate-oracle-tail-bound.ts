import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { finiteActionMenu, observableTree, solveCrra, type ActionMenu, type ForecastTree, type SolverConfig } from '../packages/bot-algo/src/oracle-crra-policy.js';
import { type OracleRules, type OracleState } from '../packages/bot-algo/src/oracle-execution.js';
import { quotePrefixUpper, relativeTailEnvelope, necessaryPerTickDepth } from '../packages/bot-algo/src/oracle-tail-bound.js';
import { ZERO, ONE, rat, add, sub, mul, div, toNumber, serializeRat, isZero, lt, type Rational } from '../packages/bot-algo/src/oracle-rational.js';

async function main() {
  const root = process.argv[2];
  if (!root) throw new Error('Usage: evaluate-oracle-tail-bound.ts <new experiment directory>');
  try { await readFile(`${root}/report.json`); throw new Error('Completed experiment exists'); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  const plan = JSON.parse(await readFile(`${root}/plan.json`, 'utf8'));
  for (const [source, expected] of Object.entries(plan.sourceHashes))
    assert.equal(createHash('sha256').update(await readFile(source)).digest('hex'), expected, `Source changed: ${source}`);
  const forecasts: readonly { readonly origin: number; readonly timestamp: number; readonly horizonSeconds: number;
    readonly stageOffsetsSeconds: readonly number[]; readonly price: number; readonly tree: ForecastTree }[] =
    JSON.parse(await readFile(`${root}/forecasts.json`, 'utf8'));
  const rules: OracleRules = { makerFee: rat('0.0002'), takerFee: rat('0.0005'), reserveLeverage: rat(2),
    entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
    quoteMinimum: rat(5), quoteMaximum: rat(10000), priceStep: rat('0.01'), maxOrders: 1 };
  const rawMenu = finiteActionMenu({ exposureTargets: [-2,-1,0,1,2].map(rat),
    limitNotionalFractions: [rat('0.25')], limitOffsetsInTicks: [1], maximumNewOrders: 1 });
  const menu: ActionMenu = (s,t,r) => rawMenu(s,t,r).filter(x => isZero(x.action.marketQuote) || !x.action.place.length);
  const config: SolverConfig = { riskAversion: 0, rules, menu, funding: () => ONE };
  const rows: unknown[] = [];
  const tolerance = rat('0.001'), initial = rat(plan.initialCash), total = plan.fullHorizonSeconds;
  const number = (q: Rational) => ({ exact: serializeRat(q), decimal: toNumber(q) });
  for (const row of forecasts) {
    const start = performance.now();
    const state: OracleState = { quote: initial, asset: ZERO, price: rat(row.price), previousPrice: rat(row.price), time: 0, orders: [] };
    const tree = observableTree(row.tree, rules.priceStep);
    const decision = solveCrra(state, tree, config, false);
    assert(decision.value !== null);
    const q = add(decision.value, ONE); // gamma=0 utility is wealth minus one
    const u = quotePrefixUpper(state, tree, config, decision.action);
    assert(u !== null && !lt(u,q), 'Prefix envelope must dominate same-action closed value');
    const remaining = total-row.horizonSeconds;
    const eps = relativeTailEnvelope(q,u,remaining,rules.quoteMaximum);
    const closedFormUpper = add(initial, mul(rat(total+1), rules.quoteMaximum));
    const result = { origin: row.origin, timestamp: row.timestamp, horizonSeconds: row.horizonSeconds,
      stageOffsetsSeconds: row.stageOffsetsSeconds, evaluatedDecisionStages: row.stageOffsetsSeconds.length,
      omittedPerSecondDecisions: remaining, selectedAction: decision.id, marketQuote: serializeRat(decision.action.marketQuote),
      prefixValue: number(q), prefixUpper: number(u), prefixRelaxationGap: number(sub(u,q)),
      tailCapAmount: number(mul(rat(remaining),rules.quoteMaximum)),
      cashEnvelopeRelativeBound: number(eps), cashEnvelopePercent: 100*toNumber(eps),
      closedFormRelativeBound: number(div(sub(closedFormUpper,q),q)),
      truncationBound: remaining === 0 ? { exact: '0', decimal: 0 } : number(eps),
      belowTolerance: remaining === 0 || lt(eps,tolerance),
      fullHorizonIdentityUsed: remaining === 0,
      scope: 'finite prefix menus/law; tail capacity counted per second; not a global learned-law certificate',
      seconds: (performance.now()-start)/1000 };
    rows.push(result);
    console.log(JSON.stringify({origin:result.origin,H:row.horizonSeconds,Q:toNumber(q),U:toNumber(u),
      epsilon:toNumber(eps),belowTolerance:result.belowTolerance,seconds:result.seconds}));
  }
  // For this horizon-query family there are <=3 prefix decisions at every H.
  // The cash theorem bounds Q_H by cash+(3+1)M, independent of the predicted law.
  // With any positive integer tail r, epsilon >= M/Q_H >= M/[cash+4M].
  const maximalPrefixWealth = add(initial,mul(rat(plan.maximumPrefixDecisions+1),rules.quoteMaximum));
  const minimumEnvelopeWithOneOmittedSecond = div(rules.quoteMaximum,maximalPrefixWealth);
  assert(!lt(minimumEnvelopeWithOneOmittedSecond,tolerance));
  const report = { tolerance: number(tolerance), fullHorizonSeconds:total,
    currentPlannerDepthResult: { smallestWholeSecondHorizonPassingTruncationTest:total,
      noPartialDepthCanPass: true, maximalPrefixWealth: number(maximalPrefixWealth),
      minimumPossibleEnvelopeWithAnyOmittedSecond:number(minimumEnvelopeWithOneOmittedSecond),
      reason:'At most three prefix decisions; Q_H <= initialCash+4M, so r>=1 implies the cash envelope >=20%. At H=T the tail is empty and its contribution is exactly zero.' },
    perSecondUnrestrictedNecessaryDepth: { seconds:necessaryPerTickDepth(total,initial,rules.quoteMaximum,tolerance),
      sufficient:false, reason:'Uses maximal possible Q_H and zero prefix gap; this is only a necessary depth for this certificate.' },
    numericalOptimalityScope:plan.precisionScope, rows };
  await writeFile(`${root}/report.json`,JSON.stringify(report,null,2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
