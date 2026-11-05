# Forecast loss plus realized Oracle wealth

The proposed objective was trained and evaluated. **Validation selected cash.**
Adding realized wealth reduced exposure and losses, but did not produce a
profitable test policy. Updating the forecast jointly did not beat keeping the
forecast frozen and training only the controller. The retained forecast/model
configuration has not been replaced.

This is an experiment with a learned numerical controller, not an executable
implementation of the noncomputable continuous Bellman selector in Lean.
It tests the proposed learning objective within explicit execution restrictions.

## Model, objective and execution

Start from the fresh direct59 model, the strongest standalone distribution
predictor from the previous comparison. Its native feature predictor stays
frozen; the trainable part is its width-128 full Gaussian-CDF knot probability
head, including its exact-zero atom. This trial does not fine-tune the entire
four-model ensemble or introduce new online calibration.

A small controller receives all predicted knot masses and zero probability,
current exposure, equity relative to initial equity, and remaining episode
fraction. A 16-unit tanh forecast encoder plus an affine state readout chooses
a continuous long/cash target through a sigmoid. Every arm starts from the same
controller, initially targeting about 4.7% exposure. This is an initialization,
not an exposure floor; requests below the actual order minimum execute zero.
Cash is also an explicit validation-selection baseline.

The loss is

\[
L = -\operatorname{mean}\log P_\theta(Y)
  +\frac{\operatorname{mean}(\widehat{\mathbb E}[Y]-Y)^2}
          {\operatorname{mean}_{\mathrm{train}}Y^2}
  -\lambda\frac{10^4}{10}
       \operatorname{mean}\left(\frac{W_T}{10000}-1\right).
\]

Here `Y` is the next-second log return and `W_T` is **realized** terminal wealth
on the training price path. Forecast-only learning preserves the base head's
NLL plus normalized log-MSE objective. The wealth term's scale is 10 bps of
episode return. It does not reward predicted advantages, confidence, turnover,
or trading frequency.

Trading uses Oracle's received-amount fees with 5 bps per side, initial wealth
10,000, actual quote minimum 5 and per-decision maximum 10,000. Targets are
continuous in `[0,1]`; there are no pending limit orders, funding is one, and
maintenance bounds are `[0,1]`. All observed prices are positive and checked
against the 0.01 price grid. Under these unleveraged restrictions, intermediate
marks cannot cause liquidation, as covered by the existing Oracle hold result.

The fee-aware target request is capped by available quote/inventory and the
aggregate trade maximum; sub-minimum requests execute zero. Holdings carry
between all decisions. One valid nonovershooting capped close occurs at the
terminal mark; any remaining inventory is marked explicitly. There is no
intermediate forced roundtrip. Funding/liquidation are inactive under the
declared configuration, rather than replaced with a different cost model.

Autograd differentiates the piecewise account execution and complete episode
wealth through the account state and forecast/controller. It does not detach
the holdings or use a straight-through gradient at order-size boundaries.
The learned controller is an approximation; neither its training nor its
continuous targets imply Bellman optimality.

## Data and selection

Frozen native predictions provide 125,293 training examples; 125,184 form 489
complete episodes. Each episode has 256 decisions spaced four seconds apart,
with terminal closing one second after the last issue: **1,021 elapsed seconds**.
The probability head still predicts one-second returns. Subsequent realized
prices are used only for training targets/account outcomes, not current inputs.

Each trainable arm runs 12 epochs with matched initialization and episode
shuffles. Adam learning rates are 0.0001 for the head and 0.01 for the
controller, with 64 episodes per batch. The first chronological validation
half selects each checkpoint by its own combined objective, including epoch
zero. The second half selects the arm by mean episode wealth, with cash as a
candidate. Each half contributes 67 complete episodes; incomplete tails are
excluded from episode scoring.

The selection is saved and hashed before loading test data. The test contains
36,134 forecasts, including 15,505 nonzero returns, and 141 complete episodes.
A separate continuous replay uses every forecast and one shared account from
**2026-08-10 00:00:01 UTC to 2026-08-11 16:08:54 UTC**. It closes only at that
full interval's endpoint. Intermediate one-second marks are retained for
drawdown and execution auditing. This longer replay is a diagnostic; it does
not change validation selection or the shorter training horizon.

The historical test and validation have been used in prior model development.
These results are exploratory, not a newly untouched holdout. The price tape
is BTCUSDT spot with the declared synthetic Oracle account rules, rather than
a historical USD-M funding/order-book simulation.

## Results

Returns below include execution costs. A basis point is 0.01%; episode figures
are arithmetic mean episode returns, not compounded returns over the test.

| Training arm | Selected epoch | Tuning mean episode, bps | Test mean episode, bps | Full test return | Full test trades |
|---|---:|---:|---:|---:|---:|
| New controller initialization, unchanged forecast | 0 | -0.485425 | -0.538250 | -0.100174% | 4 |
| Forecast-only fine-tuning, controller unchanged | 2 | -0.485425 | -0.538250 | -0.100174% | 4 |
| Wealth-trained controller, forecast frozen | 8 | 0.000000 | -0.000034 | -0.000027% | 2 |
| Joint training, lambda 0.1 | 2 | -0.201295 | -0.216197 | -0.063319% | 935 |
| Joint training, lambda 1 | 4 | -0.024468 | -0.026466 | -0.004991% | 5 |
| Joint training, lambda 10 | 8 | -0.000219 | -0.000425 | -0.000490% | 1 |
| **Cash baseline, selected on validation** | — | **0.000000** | **0.000000** | **0.000000%** | **0** |

The first row is the initialization of this new learned controller. It is
**not** the performance of the earlier endpoint optimizer or deployed bot.
The frozen-forecast trained controller is the important control for isolating
whether joint forecast updates help.

The high-wealth-weight arm ends with quote **9,994.992907** and asset
**0.000077967934**, worth approximately **4.958116** at the terminal mark.
The position's valid-close proceeds fall below the minimum, so it remains in
the account. Total wealth is **9,999.951023**, not a falsely reported cash close.
Its one executed entry has only about 5 quote units of size. The near-zero
losses are therefore mostly the consequence of near-zero exposure.

Average exposures in the continuous replay are 4.7416% for initialization,
0.000642% for the frozen-forecast trained controller, and respectively
1.6846%, 0.2249%, and 0.003242% for the three joint arms.

## Forecast quality after training

All-tick and nonzero MSE skills use the same deployed expected log return;
nonzero scoring does not renormalize away the zero probability. Full-law NLL
is rescored in float64 using the original knot density and Jacobian, matching
the previous refit evaluation.

| Forecast | All-tick log MSE skill | Nonzero log MSE skill | Full-law NLL |
|---|---:|---:|---:|
| Frozen original / controller-only arm | **4.5298%** | **6.3815%** | **-6.029284** |
| Forecast-only fine-tuning | 4.5023% | 6.3110% | -6.024415 |
| Joint, lambda 0.1 | 4.5025% | 6.3105% | -6.024411 |
| Joint, lambda 1 | 4.5104% | 6.3456% | -6.017945 |
| Joint, lambda 10 | 4.5065% | 6.3337% | -6.008359 |

Joint lambda 1 slightly improves on this run's forecast-only fine-tuning in
MSE, but every fine-tuned head remains below the original frozen head. There
is no evidence here that the wealth objective extracted a useful additional
forecast signal. This limited trial does not prove that other controllers,
horizons, inputs or optimization methods cannot succeed.

## Verification and artifacts

Seven focused tests pass: agreement with the independent existing Oracle
executor, no intermediate forced closes, permitted complete inactivity,
future-price causality for the first decision, finite-difference wealth
gradients, capped closing with residuals, and the correct gradient direction
on synthetic profitable/unprofitable paths.

All six saved variants additionally pass a complete episode comparison between
Torch execution and the independent scalar Oracle replay. Full replay traces
are reproduced and checked for valid quote-order sizes. The final audit checks
all pre-trade, post-trade and intermediate one-second marks, and confirms that
float64 rescoring reproduces the archived frozen model's means and likelihoods.

The initial diagnostic used float32 likelihood evaluation around very narrow
central knots; the final report uses the matched float64 scoring. The audit
also includes pre-trade marks in drawdown. Neither change alters training,
selection, actions or terminal wealth. The original training source and
pre-audit evaluation are preserved as experiment provenance; the trainer now
uses the corrected final scoring directly.

- `ml/oracle_wealth_training.py`: trainable controller and account rollouts.
- `ml/train_oracle_wealth_forecast.py`: matched training and held-out evaluation.
- `ml/audit_oracle_wealth_training.py`: final numerical scoring/execution audit.
- `ml/test_oracle_wealth_training.py`: focused tests.
- `data/backtests/oracle-wealth-training/plan.json`, `selection.json`,
  `evaluation.json`: protocol, frozen choices, metrics and source hashes.
- The same directory holds all six saved checkpoints, learning histories,
  actual price caches, full replay traces and scored forecast arrays.
- `data/backtests/oracle-wealth-training-run.log` and
  `data/backtests/oracle-wealth-training-audit.log`: completed logs.

The experiment took approximately 518 seconds before the final numerical audit.
No live trading configuration or existing checkpoint was overwritten.
