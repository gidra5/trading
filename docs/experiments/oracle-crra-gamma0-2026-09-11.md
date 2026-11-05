# Gamma zero comparison

Gamma zero is now the default in the Oracle bot factory and replay commands.
For nonnegative terminal wealth its CRRA utility is `U_0(W) = W - 1`, so the
policy maximizes expected terminal wealth. Zero wealth has finite utility -1;
it does not receive gamma two's negative-infinity penalty. Negative wealth
remains outside the CRRA domain. Execution costs and constraints are unchanged.

This makes the objective risk neutral. It can choose a risky trade that gamma
two rejects, but does not guarantee more trades or greater exposure in every
state. A controlled Oracle test with equally likely next prices 200 and 50
from current price 100 confirms that gamma zero enters while gamma two stays
in cash, after the same execution fees and terminal close.

## Historical results

The original validation-selected GLU forecasts, observed prices, fee rules,
action menus, starting wealth and evaluation intervals were reused without
retraining. The exact evaluated forecast hashes match the gamma-two reports.

| Evaluation | Gamma | Terminal wealth | Return | Market trades | Limit placements/fills |
| --- | ---: | ---: | ---: | ---: | ---: |
| First 3,600 ticks, two-step planning | 2 | 10,000 | 0% | 0 | 0 / 0 |
| First 3,600 ticks, two-step planning | 0 | 10,000 | 0% | 0 | 0 / 0 |
| All 151,732 test ticks, one-step planning | 2 | 10,000 | 0% | 0 | 0 / 0 |
| All 151,732 test ticks, one-step planning | 0 | 10,000 | 0% | 0 | 0 / 0 |

All four runs have zero drawdown and liquidation. The gamma-zero two-step run
took 266.0 seconds; the full one-step run took 178.5 seconds. The matching
buy-and-hold results remain +0.029325% over the first hour and -2.165959% over
the full held-out interval after the same execution fees.

**The bot did not become more aggressive in these replays.** Gamma two's risk
aversion was not necessary to keep it in cash: no available trade beat cash
under the gamma-zero objective either. These results concern the supplied
very short forecast trees and finite menus, not every holding horizon or the
unrestricted market optimum. The tape and account assumptions remain the
synthetic Oracle replay described in the
[original implementation report](oracle-crra-implementation-2026-09-11.md).

Reports:

- `data/backtests/stochastic-crra/first-hour-gamma0/report.json`
- `data/backtests/stochastic-crra/full-test-horizon1-gamma0/report.json`

## Implementation and proof

`SolverConfig.riskAversion` explicitly selects 0 or 2. The bot factory defaults
to zero; use `riskAversion: 2` for the earlier objective. Other gamma values
are rejected rather than evaluated through an unverified float approximation.
Gamma two keeps its previous unnormalized `-1/W` score and proven affine
correspondence to normalized CRRA.

Every new decision certificate records its risk aversion. The Lean checker
uses the matching utility and rejects a change in risk aversion between parent
and child decisions. The original certificates that omit this field retain
their defined gamma-two interpretation.

The Lean `RiskAversion` domain now includes zero. `ExactRiskNeutral.lean`
proves gamma-zero utility, its finite domain, rational-to-real CRRA refinement,
finite utility at zero wealth, and the expected-wealth identity. Existing
exact expectation, selection and Bellman theorems apply to these scores.
The same finite-tree/menu and execution-refinement assumptions still apply;
this does not turn a sampled tree into a proof for the continuous learned law.

Verification passed: 1,082 Lean theorem audits, 41 TypeScript tests, full
server typechecking, and six certificate regressions including four expected
rejections. The 108 Oracle execution fixtures and three gamma-zero replay
certificates pass the checker. The three original gamma-two certificates
also still pass. Results are recorded in
`data/backtests/stochastic-crra/lean-verification-gamma0.json` and
`data/backtests/stochastic-crra/lean-verification-gamma2.json`.

Reproduce:

```sh
npm run backtest:oracle-crra -- --gamma 0
npm run backtest:oracle-crra -- --gamma 0 --forecasts data/backtests/stochastic-crra/test-full-3bins.jsonl --count 151732 --horizon 1 --certificates 0 --output data/backtests/stochastic-crra/full-test-horizon1-gamma0
npm run test:oracle-crra
npm run typecheck -w @trading/server
npm run theory:check
npm run test:oracle-crra:proof
npm run verify:oracle-crra
```

Use `--gamma 2` explicitly to rerun the earlier utility. `experiment:oracle-crra`
also accepts `--gamma` and defaults to zero.
