import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { logInterval } from '../packages/bot-algo/src/oracle-log-policy.js';
import { ONE, ZERO, add, div, lt, mul, rat, serializeRat, sub, toNumber, type Rational } from '../packages/bot-algo/src/oracle-rational.js';

// This evaluator applies the distribution-independent EVERY-tick theorem.
// Validation residuals and realized replay wealth are never substituted for
// uniform Bellman inequalities or optimal values under the stochastic law.
async function main() {
  const replay = process.argv[2], output = process.argv[3];
  if (!replay || !output) throw new Error('Usage: evaluate-oracle-continuation-bound.ts <replay-directory> <new-output-directory>');
  const plan = JSON.parse(await readFile(`${replay}/plan.json`, 'utf8'));
  const performance = JSON.parse(await readFile(`${replay}/report.json`, 'utf8'));
  const execution = JSON.parse(await readFile(`${replay}/exact-replay.json`, 'utf8'));
  assert.equal(plan.gamma, 1);
  assert.equal(plan.decisionStrideSeconds, 1);
  assert.equal(plan.terminalRequirement, 'zero-asset-and-no-orders');
  assert(execution.complete && execution.allActualActionsExactlyFeasible);
  assert.equal(performance.decisions, execution.decisions);
  const H: number = plan.rollingHorizonSeconds, T = 48 * 3600;
  assert(Number.isInteger(H) && H > 0 && H < T);
  // The replay fixture begins at $10,000 cash; the independent cash policy is
  // feasible under every price law with these rules and zero cash carry.
  const initial = rat(10000), M = rat(plan.trainingMetadata.rules.maximum);
  const epsilon = rat('0.001');
  const fullUpper = add(initial, mul(rat(T + 1), M));
  const relativeUpper = sub(div(fullUpper, initial), ONE);
  const threshold = logInterval(add(ONE, epsilon));
  const residualThreshold = div(threshold.lower, rat(2 * H));
  const number = (x: Rational) => ({ exact: serializeRat(x), decimal: toNumber(x) });
  const sourcePaths = [
    'scripts/evaluate-oracle-continuation-bound.ts',
    'docs/theory/lean/TradingTheory/OracleContinuationValue.lean',
    'packages/bot-algo/src/oracle-log-policy.ts',
  ];
  const sourceHashes = Object.fromEntries(await Promise.all(sourcePaths.map(async path => [path,
    createHash('sha256').update(await readFile(path)).digest('hex')])));
  const validation = plan.trainingMetadata.validation;
  const report = {
    replay, sourceHashes, gamma: 1, forecastStepSeconds: 1,
    checkpointSha256: plan.trainingMetadata.checkpointSha256,
    valueSha256: plan.valueSha256,
    horizonTruncation: {
      definition: 'exp(V_48h - V_H) - 1, comparing optimal flat-ending policies',
      scope: 'All feasible continuous Oracle controls with the aggregate cap at EVERY tick; no law approximation required for this upper bound',
      initialCash: number(initial), aggregateQuoteCap: number(M), fullHorizonSeconds: T,
      fullTerminalWealthAndCeUpper: number(fullUpper),
      certifiedPrefixCeLower: number(initial),
      prefixLowerPolicy: 'Stay cash; not a claim that the one-hour optimum is cash',
      checkpoints: [H, 24 * 3600, T - 1, T].map(h => {
        const upper = h === T ? ZERO : relativeUpper;
        return { horizonSeconds: h, relativeUpper: number(upper),
          relativeUpperPercent: toNumber(upper) * 100, belowToleranceCertified: lt(upper, epsilon),
          fullHorizonIdentityUsed: h === T, earlierFailureEstablished: false };
      }),
      firstCertifiedCheckpointSeconds: T,
      earliestSufficientDepthIdentified: false,
      oneHourRetains95PercentCertified: false,
      limitation: 'The universal cap envelope is very loose. It does not establish that the actual tail is large, or that an earlier cutoff fails.',
    },
    continuationApproximation: {
      horizonSeconds: H, targetRelativeCeRegret: number(epsilon),
      logRegretUpperFormula: '2 * horizon * uniform Bellman residual',
      sufficientStrictUniformResidualThreshold: number(residualThreshold),
      thresholdScope: 'Finite-log-value domain, exact terminal anchor, every feasible competing action and chosen-policy lower backup, at all reachable states and remaining horizons; distribution and menu errors must also be covered',
      validationFiniteAccounts: validation.validationFinite,
      validationTotalAccounts: 96,
      validationConditionalSamples: 32,
      validationMeanAbsResidual: validation.validationMeanAbsResidual,
      validationMaxAbsResidual: validation.validationMaxAbsResidual,
      observedMaximumToRequiredThreshold: validation.validationMaxAbsResidual / toNumber(residualThreshold),
      uniformResidualCertificate: null,
      learnedPolicyCeRegretCertificate: null,
      validationIsNotUniformCertificate: true,
    },
    performance, execution,
  };
  await mkdir(output);
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ fullCeUpper: toNumber(fullUpper), oneHourRelativeUpperPercent: toNumber(relativeUpper) * 100,
    firstCertifiedCheckpointSeconds: T, strictUniformResidualThreshold: toNumber(residualThreshold),
    observedValidationMaxResidual: validation.validationMaxAbsResidual, uniformCertificate: false }, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
