# Live fast-feature early screen — 2026-08-19

Generated at 2026-08-19T18:25:44.575Z. Target coverage is **30.933 observed hours**.

## Interpretation

- Degraded compact books: krakenSpot (71.6% valid). Treat their features as missing behind observed/age masks.
- 39 feature/horizon pairs clear the strict early-effect rule; they are candidates for the 3-7 day confirmation, not production promotion.
- 47 feature/horizon pairs are consistently negative in this window, but one day is not enough for permanent rejection.
- Repeat after 3 and 7 observed days. Promote only effects that retain positive held-out gain across separated days and in a joint ablation against the established basis.
- This is explicitly an early effect and broken-feed screen. It does not promote or permanently reject model inputs.

## Causal validation

- Every fast feature bucket from second t first becomes available at t+1; targets begin from the completed price at t.
- BTCUSDT end-of-completed-second log return at 1s, 5s, 15s, and 60s, aligned by local receive time
- Categorical conditional density from prior same-horizon return and trailing absolute-return state.
- Baseline plus one feature discretized into training-only tertiles; target quartiles are also training-only.
- Inactivity; sign conditional on an active return; active-magnitude threshold events at training-only 25/50/75/90% quantiles; sign inside small/large magnitude subsets; magnitude-tail probability within each sign; and a joint zero/sign/magnitude state.
- First 60% trains; final 40% evaluates in six chronological blocks. Three deterministic feature permutations estimate finite-sample/extra-state bias.
- One live day is an early effect/broken-feed screen, not promotion or rejection evidence. Scores are exploratory and not multiple-testing adjusted.

## Feed health

Compact book rows: 76,685; observed coverage: 21.387h; maximum gap: 14s.

| venue | valid rows | valid fraction | median age | p99 age | status |
|---|---:|---:|---:|---:|---|
| binanceSpot | 76,524 | 99.79% | 61.0ms | 205.8ms | healthy |
| coinbaseSpot | 76,679 | 99.99% | 35.0ms | 104.0ms | healthy |
| krakenSpot | 54,888 | 71.58% | 54.0ms | 735.0ms | degraded |
| deribitPerpetual | 76,674 | 99.99% | 171.0ms | 390.3ms | healthy |
| binancePerpetual | 76,590 | 99.88% | 36.0ms | 455.0ms | healthy |

Kraken-derived scores from this archive are marked `feed-contaminated` and excluded from all rankings. The collector failed to truncate the reconstructed book to the subscribed depth; [Kraken's reconstruction rules](https://docs.kraken.com/exchange/guides/websockets/book-checksum-v2) explicitly say that zero-quantity removals are not sent for levels that merely fall out of scope. The collector now truncates to depth 100 and reconnects for a fresh snapshot whenever the reconstructed book crosses. Historical compact rows are not rewritten.

All-market liquidation messages: 37,745; BTCUSDT messages: 1,357 (3.60%). Absence is encoded as zero rather than dropping non-event seconds.

Deribit perpetual trades: 98,656; option trades: 14,325.

## Family summary

| family | evaluated pairs | strict early effects | weak in window | best clean feature | target | bits/target | lower block bound |
|---|---:|---:|---:|---|---:|---:|---:|
| cross-exchange-book | 160 | 17 | 28 | binance_perpetual_basis_bps_mean_60s | 15s | 0.076402 | 0.002256 |
| btc-liquidations | 48 | 0 | 4 | btc_liquidation_count_60s | 15s | 0.028223 | -0.005696 |
| deribit-perpetual-flow | 48 | 21 | 1 | deribit_perpetual_trade_count_60s | 60s | 0.050320 | 0.005351 |
| deribit-option-flow | 48 | 1 | 14 | deribit_option_trade_count_5s | 1s | 0.017734 | -0.003741 |

Whole-return strict rows by family: cross-exchange-book 17/160; btc-liquidations 0/48; deribit-perpetual-flow 21/48; deribit-option-flow 1/48. The main pattern is still magnitude/activity-state information rather than stable direction; the component audit below makes that distinction explicit.

## Separate future-component targets

Each row below selects the best individually tested input for that target component. `P(|R| >= threshold)` and `P(|R| < threshold)` are complementary binary targets and therefore have the same information score at the same threshold; only the `>=` form is listed. Thresholds are active-return quantiles fitted on the training interval and are shown in bps. A conditional row is scored only on outcomes satisfying its condition.

| horizon | component | threshold | tested inputs | strict effects | best family / input | bits/eligible target | lower block bound | status |
|---:|---|---:|---:|---:|---|---:|---:|---|
| 1s | P(inactive) | n/a | 76 | 8 | cross-exchange-book / binance_depth_churn_log_quote | 0.020951 | 0.001622 | large-early-effect |
| 1s | P(zero/sign/magnitude quartile) | n/a | 76 | 16 | cross-exchange-book / binance_spot_spread_bps | 0.264401 | 0.058455 | large-early-effect |
| 1s | P(|R| >= Q25 | active) | 0.0015 bps | 76 | 2 | cross-exchange-book / coinbase_spot_l1_imbalance_mean_15s | 0.013805 | 0.001600 | large-early-effect |
| 1s | P(|R| >= Q50 | active) | 0.0016 bps | 76 | 0 | deribit-perpetual-flow / deribit_perpetual_trade_count_1s | 0.026051 | -0.008560 | inconclusive |
| 1s | P(|R| >= Q50 | negative) | 0.0016 bps | 76 | 0 | cross-exchange-book / binance_spot_l1_imbalance | 0.037791 | -0.039299 | inconclusive |
| 1s | P(|R| >= Q50 | positive) | 0.0016 bps | 76 | 0 | deribit-option-flow / deribit_option_trade_count_15s | 0.013743 | -0.012474 | inconclusive |
| 1s | P(|R| >= Q75 | active) | 0.0016 bps | 76 | 7 | deribit-perpetual-flow / deribit_perpetual_trade_amount_5s | 0.078153 | 0.000934 | large-early-effect |
| 1s | P(|R| >= Q75 | negative) | 0.0016 bps | 76 | 9 | deribit-perpetual-flow / deribit_perpetual_trade_amount_15s | 0.073418 | 0.003343 | large-early-effect |
| 1s | P(|R| >= Q75 | positive) | 0.0016 bps | 76 | 6 | deribit-perpetual-flow / deribit_perpetual_trade_count_15s | 0.049700 | 0.005094 | large-early-effect |
| 1s | P(|R| >= Q90 | active) | 0.2852 bps | 76 | 25 | deribit-perpetual-flow / deribit_perpetual_trade_amount_5s | 0.098163 | 0.005674 | large-early-effect |
| 1s | P(|R| >= Q90 | negative) | 0.2852 bps | 76 | 13 | deribit-perpetual-flow / deribit_perpetual_trade_amount_15s | 0.123249 | 0.005759 | large-early-effect |
| 1s | P(|R| >= Q90 | positive) | 0.2852 bps | 76 | 12 | deribit-perpetual-flow / deribit_perpetual_trade_count_15s | 0.065074 | 0.007735 | large-early-effect |
| 1s | P(positive | active) | n/a | 76 | 19 | cross-exchange-book / binance_depth_churn_log_quote | 0.022469 | 0.005611 | large-early-effect |
| 1s | P(positive | active, |R| >= Q50) | 0.0016 bps | 76 | 13 | cross-exchange-book / spot_mid_dispersion_bps_mean_60s | 0.023596 | 0.001404 | large-early-effect |
| 1s | P(positive | active, |R| >= Q75) | 0.0016 bps | 76 | 3 | btc-liquidations / btc_liquidation_count_60s | 0.006488 | 0.000972 | large-early-effect |
| 1s | P(positive | active, |R| >= Q90) | 0.2864 bps | 76 | 2 | cross-exchange-book / spot_mid_dispersion_bps | 0.009891 | 0.001989 | large-early-effect |
| 1s | P(positive | active, |R| < Q25) | 0.0015 bps | 76 | 2 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.001242 | 0.000690 | large-early-effect |
| 1s | P(positive | active, |R| < Q50) | 0.0016 bps | 76 | 1 | cross-exchange-book / binance_depth_churn_log_quote | 0.007704 | 0.001355 | large-early-effect |
| 1s | P(positive | active, |R| < Q75) | 0.0016 bps | 76 | 3 | cross-exchange-book / binance_depth_churn_log_quote | 0.010581 | 0.001629 | large-early-effect |
| 5s | P(inactive) | n/a | 76 | 17 | cross-exchange-book / binance_depth_churn_log_quote | 0.028312 | 0.004931 | large-early-effect |
| 5s | P(zero/sign/magnitude quartile) | n/a | 76 | 10 | deribit-perpetual-flow / deribit_perpetual_trade_count_15s | 0.068388 | 0.009031 | large-early-effect |
| 5s | P(|R| >= Q25 | active) | 0.0016 bps | 76 | 5 | cross-exchange-book / binance_depth_churn_log_quote | 0.012100 | 0.001795 | large-early-effect |
| 5s | P(|R| >= Q50 | active) | 0.0016 bps | 76 | 16 | deribit-perpetual-flow / deribit_perpetual_trade_amount_5s | 0.059118 | 0.014424 | large-early-effect |
| 5s | P(|R| >= Q50 | negative) | 0.0016 bps | 76 | 18 | deribit-perpetual-flow / deribit_perpetual_trade_amount_15s | 0.065491 | 0.014107 | large-early-effect |
| 5s | P(|R| >= Q50 | positive) | 0.0016 bps | 76 | 18 | deribit-perpetual-flow / deribit_perpetual_trade_count_15s | 0.062595 | 0.019987 | large-early-effect |
| 5s | P(|R| >= Q75 | active) | 0.3117 bps | 76 | 32 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.092122 | 0.027089 | large-early-effect |
| 5s | P(|R| >= Q75 | negative) | 0.3117 bps | 76 | 38 | cross-exchange-book / binance_perpetual_basis_bps | 0.135303 | 0.036197 | large-early-effect |
| 5s | P(|R| >= Q75 | positive) | 0.3117 bps | 76 | 27 | cross-exchange-book / spot_mid_dispersion_bps | 0.109414 | 0.010220 | large-early-effect |
| 5s | P(|R| >= Q90 | active) | 0.9282 bps | 76 | 34 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.104841 | 0.027999 | large-early-effect |
| 5s | P(|R| >= Q90 | negative) | 0.9282 bps | 76 | 39 | cross-exchange-book / spot_mid_dispersion_bps_mean_60s | 0.130825 | 0.021789 | large-early-effect |
| 5s | P(|R| >= Q90 | positive) | 0.9282 bps | 76 | 19 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.121720 | 0.025431 | large-early-effect |
| 5s | P(positive | active) | n/a | 76 | 1 | cross-exchange-book / binance_remove_imbalance | 0.002162 | 0.000704 | large-early-effect |
| 5s | P(positive | active, |R| >= Q50) | 0.0016 bps | 76 | 9 | deribit-option-flow / deribit_option_trade_amount_60s | 0.023960 | 0.008776 | large-early-effect |
| 5s | P(positive | active, |R| >= Q75) | 0.3117 bps | 76 | 10 | cross-exchange-book / best_executable_spread_bps | 0.036387 | 0.018821 | large-early-effect |
| 5s | P(positive | active, |R| >= Q90) | 0.9283 bps | 76 | 10 | cross-exchange-book / spot_mid_dispersion_bps_mean_15s | 0.029268 | 0.006598 | large-early-effect |
| 5s | P(positive | active, |R| < Q25) | 0.0016 bps | 76 | 10 | deribit-perpetual-flow / deribit_perpetual_trade_count_5s | 0.017538 | 0.008128 | large-early-effect |
| 5s | P(positive | active, |R| < Q50) | 0.0016 bps | 76 | 9 | deribit-perpetual-flow / deribit_perpetual_trade_count_5s | 0.016966 | 0.010628 | large-early-effect |
| 5s | P(positive | active, |R| < Q75) | 0.3116 bps | 76 | 7 | deribit-perpetual-flow / deribit_perpetual_trade_count_5s | 0.015237 | 0.003715 | large-early-effect |
| 15s | P(inactive) | n/a | 76 | 19 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.037659 | 0.004816 | large-early-effect |
| 15s | P(zero/sign/magnitude quartile) | n/a | 76 | 18 | cross-exchange-book / binance_perpetual_spread_bps | 0.106490 | 0.078418 | large-early-effect |
| 15s | P(|R| >= Q25 | active) | 0.0016 bps | 76 | 16 | deribit-perpetual-flow / deribit_perpetual_trade_amount_5s | 0.025442 | 0.008253 | large-early-effect |
| 15s | P(|R| >= Q50 | active) | 0.3016 bps | 76 | 32 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.072978 | 0.020438 | large-early-effect |
| 15s | P(|R| >= Q50 | negative) | 0.3016 bps | 76 | 27 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.081601 | 0.018707 | large-early-effect |
| 15s | P(|R| >= Q50 | positive) | 0.3016 bps | 76 | 28 | cross-exchange-book / best_executable_spread_bps | 0.076691 | 0.014437 | large-early-effect |
| 15s | P(|R| >= Q75 | active) | 0.9317 bps | 76 | 40 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.101462 | 0.028208 | large-early-effect |
| 15s | P(|R| >= Q75 | negative) | 0.9317 bps | 76 | 33 | cross-exchange-book / spot_mid_dispersion_bps_mean_60s | 0.107186 | 0.024032 | large-early-effect |
| 15s | P(|R| >= Q75 | positive) | 0.9317 bps | 76 | 18 | cross-exchange-book / binance_perpetual_basis_bps_mean_60s | 0.095809 | 0.027969 | large-early-effect |
| 15s | P(|R| >= Q90 | active) | 1.6931 bps | 76 | 30 | btc-liquidations / btc_liquidation_count_60s | 0.111302 | 0.004475 | large-early-effect |
| 15s | P(|R| >= Q90 | negative) | 1.6931 bps | 76 | 33 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.143852 | 0.047650 | large-early-effect |
| 15s | P(|R| >= Q90 | positive) | 1.6931 bps | 76 | 15 | btc-liquidations / btc_liquidation_count_60s | 0.107374 | 0.003270 | large-early-effect |
| 15s | P(positive | active) | n/a | 76 | 8 | cross-exchange-book / binance_spot_l1_imbalance_mean_15s | 0.025671 | 0.000974 | large-early-effect |
| 15s | P(positive | active, |R| >= Q50) | 0.3016 bps | 76 | 12 | cross-exchange-book / binance_spot_l1_imbalance_mean_5s | 0.042839 | 0.012477 | large-early-effect |
| 15s | P(positive | active, |R| >= Q75) | 0.9317 bps | 76 | 3 | deribit-perpetual-flow / deribit_perpetual_trade_amount_60s | 0.018831 | 0.006626 | large-early-effect |
| 15s | P(positive | active, |R| >= Q90) | 1.6931 bps | 76 | 1 | cross-exchange-book / binance_depth_pressure_mean_5s | 0.023857 | 0.002644 | large-early-effect |
| 15s | P(positive | active, |R| < Q25) | 0.0016 bps | 76 | 1 | deribit-perpetual-flow / deribit_perpetual_trade_imbalance_60s | 0.011983 | 0.008570 | large-early-effect |
| 15s | P(positive | active, |R| < Q50) | 0.3025 bps | 76 | 1 | cross-exchange-book / binance_spot_l1_imbalance_mean_60s | 0.020274 | 0.000477 | large-early-effect |
| 15s | P(positive | active, |R| < Q75) | 0.9319 bps | 76 | 2 | cross-exchange-book / binance_spot_l1_imbalance_mean_15s | 0.035500 | 0.003649 | large-early-effect |
| 60s | P(inactive) | n/a | 76 | 4 | deribit-perpetual-flow / deribit_perpetual_trade_count_15s | 0.011390 | 0.003135 | large-early-effect |
| 60s | P(zero/sign/magnitude quartile) | n/a | 76 | 6 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.113909 | 0.018114 | large-early-effect |
| 60s | P(|R| >= Q25 | active) | 0.3802 bps | 76 | 12 | deribit-perpetual-flow / deribit_perpetual_trade_count_15s | 0.030525 | 0.008627 | large-early-effect |
| 60s | P(|R| >= Q50 | active) | 1.0838 bps | 76 | 19 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.049297 | 0.013595 | large-early-effect |
| 60s | P(|R| >= Q50 | negative) | 1.0838 bps | 76 | 16 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.059057 | 0.034307 | large-early-effect |
| 60s | P(|R| >= Q50 | positive) | 1.0838 bps | 76 | 9 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.047650 | 0.003599 | large-early-effect |
| 60s | P(|R| >= Q75 | active) | 2.0759 bps | 76 | 18 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.111362 | 0.035880 | large-early-effect |
| 60s | P(|R| >= Q75 | negative) | 2.0759 bps | 76 | 24 | deribit-perpetual-flow / deribit_perpetual_trade_count_60s | 0.155494 | 0.060402 | large-early-effect |
| 60s | P(|R| >= Q75 | positive) | 2.0759 bps | 76 | 12 | deribit-perpetual-flow / deribit_perpetual_trade_amount_60s | 0.137293 | 0.047042 | large-early-effect |
| 60s | P(|R| >= Q90 | active) | 3.3515 bps | 76 | 24 | cross-exchange-book / binance_perpetual_basis_bps_mean_15s | 0.185320 | 0.022824 | large-early-effect |
| 60s | P(|R| >= Q90 | negative) | 3.3515 bps | 76 | 22 | cross-exchange-book / binance_perpetual_basis_bps_mean_15s | 0.225153 | 0.020760 | large-early-effect |
| 60s | P(|R| >= Q90 | positive) | 3.3515 bps | 76 | 5 | deribit-perpetual-flow / deribit_perpetual_trade_amount_60s | 0.237138 | 0.051104 | large-early-effect |
| 60s | P(positive | active) | n/a | 76 | 0 | cross-exchange-book / coinbase_spot_l1_imbalance_mean_5s | 0.009581 | -0.000094 | inconclusive |
| 60s | P(positive | active, |R| >= Q50) | 1.0840 bps | 76 | 1 | cross-exchange-book / binance_depth_churn_log_quote | 0.002169 | 0.000196 | large-early-effect |
| 60s | P(positive | active, |R| >= Q75) | 2.0762 bps | 76 | 1 | cross-exchange-book / binance_depth_churn_log_quote | 0.007426 | 0.003252 | large-early-effect |
| 60s | P(positive | active, |R| >= Q90) | 3.3515 bps | 76 | 0 | cross-exchange-book / binance_depth_churn_log_quote | 0.003223 | -0.005408 | inconclusive |
| 60s | P(positive | active, |R| < Q25) | 0.3792 bps | 76 | 0 | cross-exchange-book / binance_spot_l1_imbalance_mean_15s | 0.022220 | -0.019879 | inconclusive |
| 60s | P(positive | active, |R| < Q50) | 1.0840 bps | 76 | 1 | cross-exchange-book / coinbase_spot_l1_imbalance_mean_5s | 0.005829 | 0.001243 | large-early-effect |
| 60s | P(positive | active, |R| < Q75) | 2.0762 bps | 76 | 5 | cross-exchange-book / coinbase_spot_l1_imbalance | 0.012459 | 0.000578 | large-early-effect |

The component score is conditional information per eligible outcome, so values from differently conditioned rows are not additive and should not be compared as though they used the same sample population. The joint zero/sign/magnitude row is the closest component audit to a single complete-distribution target.

At 1s, Q25/Q50/Q75 active magnitudes cluster around one BTC price tick (about 0.0015–0.0016 bps in this window), so those are not three economically distinct regimes. The Q90 row is the first clearly separated 1s tail threshold.

## Strict early-effect candidates

A row must have a positive 95% block bound, at least 5/6 positive chronological blocks, and exceed its shuffled-feature control. These remain confirmation candidates only.

| family | feature | target | bits/target | lower block bound | shuffled control | positive blocks | effective outcomes |
|---|---|---:|---:|---:|---:|---:|---:|
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 60s | 0.050320 | 0.005351 | -0.001398 | 5/6 | 500 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 60s | 0.050221 | 0.010670 | 0.002887 | 5/6 | 500 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 15s | 0.048260 | 0.012390 | 0.000744 | 6/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 15s | 0.045328 | 0.009724 | 0.003252 | 6/6 | 2,038 |
| cross-exchange-book | binance_perpetual_spread_bps | 15s | 0.044871 | 0.013351 | -0.001336 | 6/6 | 2,040 |
| cross-exchange-book | binance_spot_spread_bps | 15s | 0.042440 | 0.009403 | 0.004490 | 6/6 | 2,041 |
| cross-exchange-book | deribit_perpetual_spread_bps | 15s | 0.039276 | 0.011417 | -0.000099 | 6/6 | 2,038 |
| cross-exchange-book | binance_depth_churn_log_quote | 60s | 0.037186 | 0.010456 | 0.002443 | 6/6 | 500 |
| cross-exchange-book | coinbase_spot_spread_bps | 15s | 0.035637 | 0.017575 | -0.001998 | 6/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 15s | 0.032058 | 0.003335 | 0.003604 | 6/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 15s | 0.029868 | 0.001848 | 0.005977 | 6/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 15s | 0.028011 | 0.009345 | 0.000016 | 6/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 60s | 0.026765 | 0.004773 | 0.003327 | 5/6 | 500 |
| cross-exchange-book | binance_depth_churn_log_quote | 1s | 0.026435 | 0.001636 | 0.001332 | 6/6 | 30,608 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 5s | 0.025671 | 0.010499 | 0.001138 | 6/6 | 6,145 |
| cross-exchange-book | binance_depth_churn_log_quote | 15s | 0.025124 | 0.005188 | 0.005814 | 6/6 | 2,038 |
| cross-exchange-book | binance_depth_churn_log_quote | 5s | 0.023764 | 0.007252 | -0.002085 | 6/6 | 6,144 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 1s | 0.023209 | 0.001279 | -0.002669 | 6/6 | 30,723 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_60s | 15s | 0.021441 | 0.005820 | -0.000624 | 6/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 60s | 0.019696 | 0.003305 | 0.003454 | 5/6 | 500 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 5s | 0.019451 | 0.005396 | 0.003186 | 5/6 | 6,145 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 60s | 0.015980 | 0.001369 | 0.001594 | 5/6 | 500 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 1s | 0.013979 | 0.001053 | -0.002190 | 6/6 | 30,723 |
| cross-exchange-book | binance_spot_l1_imbalance_mean_60s | 15s | 0.013415 | 0.003912 | 0.002733 | 6/6 | 2,044 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_1s | 60s | 0.012531 | 0.000480 | 0.002529 | 5/6 | 500 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_1s | 60s | 0.012531 | 0.000480 | 0.002529 | 5/6 | 500 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_15s | 15s | 0.012073 | 0.002399 | -0.002618 | 6/6 | 2,037 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_5s | 60s | 0.011861 | 0.002403 | 0.000869 | 6/6 | 500 |
| cross-exchange-book | coinbase_spot_l1_imbalance_mean_5s | 60s | 0.011023 | 0.001294 | 0.002625 | 5/6 | 500 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 5s | 0.010936 | 0.001884 | 0.003370 | 5/6 | 6,145 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_60s | 15s | 0.010693 | 0.000252 | -0.000268 | 5/6 | 2,038 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_1s | 15s | 0.009966 | 0.000621 | -0.001492 | 6/6 | 2,038 |
| cross-exchange-book | best_executable_spread_bps | 1s | 0.009374 | 0.001232 | -0.000641 | 5/6 | 30,608 |
| cross-exchange-book | spot_mid_dispersion_bps | 1s | 0.009074 | 0.001254 | -0.000581 | 5/6 | 30,608 |
| cross-exchange-book | spot_mid_dispersion_bps_mean_5s | 1s | 0.007256 | 0.001744 | -0.000260 | 6/6 | 30,603 |
| deribit-option-flow | deribit_option_trade_amount_60s | 1s | 0.005984 | 0.000147 | 0.000552 | 5/6 | 30,701 |
| cross-exchange-book | binance_add_imbalance | 5s | 0.003968 | 0.000035 | -0.001361 | 5/6 | 6,144 |
| cross-exchange-book | binance_add_imbalance | 15s | 0.003198 | 0.001911 | -0.000325 | 6/6 | 2,038 |
| cross-exchange-book | spot_mid_dispersion_bps_mean_15s | 1s | 0.002307 | 0.000077 | -0.000200 | 5/6 | 30,588 |

## Best exploratory scores

| family | feature | target | bits/target | excess over shuffle | blocks | classification |
|---|---|---:|---:|---:|---:|---|
| cross-exchange-book | binance_perpetual_basis_bps_mean_60s | 15s | 0.076402 | 0.079033 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 60s | 0.050320 | 0.051718 | 5/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 60s | 0.050221 | 0.047334 | 5/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 15s | 0.048260 | 0.047516 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps_mean_15s | 15s | 0.047992 | 0.051612 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 15s | 0.045328 | 0.042076 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_spread_bps | 15s | 0.044871 | 0.046207 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps_mean_5s | 15s | 0.042488 | 0.043875 | 4/6 | inconclusive |
| cross-exchange-book | binance_spot_spread_bps | 15s | 0.042440 | 0.037950 | 6/6 | large-early-effect |
| cross-exchange-book | spot_mid_dispersion_bps | 15s | 0.039716 | 0.041331 | 4/6 | inconclusive |
| cross-exchange-book | best_executable_spread_bps | 15s | 0.039597 | 0.041268 | 4/6 | inconclusive |
| cross-exchange-book | deribit_perpetual_spread_bps | 15s | 0.039276 | 0.039375 | 6/6 | large-early-effect |
| cross-exchange-book | spot_mid_dispersion_bps_mean_5s | 15s | 0.038939 | 0.039623 | 4/6 | inconclusive |
| cross-exchange-book | binance_depth_churn_log_quote | 60s | 0.037186 | 0.034742 | 6/6 | large-early-effect |
| cross-exchange-book | coinbase_spot_spread_bps | 15s | 0.035637 | 0.037636 | 6/6 | large-early-effect |
| cross-exchange-book | spot_mid_dispersion_bps_mean_15s | 15s | 0.032809 | 0.035368 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 15s | 0.032058 | 0.028454 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps | 15s | 0.030923 | 0.033449 | 4/6 | inconclusive |
| cross-exchange-book | binance_perpetual_basis_bps_mean_60s | 60s | 0.030033 | 0.032911 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 15s | 0.029868 | 0.023891 | 6/6 | large-early-effect |
| btc-liquidations | btc_liquidation_count_60s | 15s | 0.028223 | 0.030115 | 4/6 | inconclusive |
| btc-liquidations | btc_liquidation_amount_60s | 15s | 0.028223 | 0.030115 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 15s | 0.028011 | 0.027996 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 60s | 0.026765 | 0.023438 | 5/6 | large-early-effect |
| cross-exchange-book | binance_depth_churn_log_quote | 1s | 0.026435 | 0.025103 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 1s | 0.026425 | 0.026407 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 5s | 0.025671 | 0.024534 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 1s | 0.025559 | 0.025604 | 5/6 | inconclusive |
| cross-exchange-book | binance_depth_churn_log_quote | 15s | 0.025124 | 0.019309 | 6/6 | large-early-effect |
| cross-exchange-book | binance_depth_churn_log_quote | 5s | 0.023764 | 0.025849 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 1s | 0.023209 | 0.025878 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_1s | 1s | 0.021798 | 0.022169 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_1s | 1s | 0.021798 | 0.022169 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_60s | 15s | 0.021441 | 0.022065 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps_mean_15s | 60s | 0.021292 | 0.023634 | 3/6 | inconclusive |
| btc-liquidations | btc_liquidation_count_60s | 60s | 0.021059 | 0.022236 | 5/6 | inconclusive |
| btc-liquidations | btc_liquidation_amount_60s | 60s | 0.021059 | 0.022236 | 5/6 | inconclusive |
| cross-exchange-book | spot_mid_dispersion_bps_mean_60s | 15s | 0.020616 | 0.023355 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_1s | 15s | 0.020488 | 0.020268 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_1s | 15s | 0.020488 | 0.020268 | 6/6 | inconclusive |

## Consistently negative rows in this window

These are candidates for later rejection, not rejected features. A single live day cannot distinguish a genuinely useless input from a regime-specific failure.

| family | feature | target | bits/target | blocks |
|---|---|---:|---:|---:|
| cross-exchange-book | binance_spot_spread_bps | 5s | -0.115484 | 1/6 |
| cross-exchange-book | binance_perpetual_spread_bps | 5s | -0.112933 | 1/6 |
| cross-exchange-book | deribit_perpetual_spread_bps | 5s | -0.106704 | 1/6 |
| cross-exchange-book | coinbase_spot_spread_bps | 5s | -0.074708 | 1/6 |
| btc-liquidations | btc_liquidation_imbalance_60s | 60s | -0.072131 | 0/6 |
| btc-liquidations | btc_liquidation_imbalance_15s | 60s | -0.044832 | 0/6 |
| btc-liquidations | btc_liquidation_imbalance_15s | 15s | -0.041138 | 0/6 |
| cross-exchange-book | binance_perpetual_basis_bps | 5s | -0.038074 | 1/6 |
| cross-exchange-book | binance_perpetual_basis_bps_mean_5s | 5s | -0.033198 | 1/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 15s | -0.032291 | 0/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 5s | -0.026697 | 1/6 |
| cross-exchange-book | binance_perpetual_basis_bps | 1s | -0.025325 | 0/6 |
| cross-exchange-book | coinbase_spot_spread_bps | 60s | -0.023882 | 1/6 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_60s | 60s | -0.022672 | 1/6 |
| cross-exchange-book | binance_perpetual_basis_bps_mean_5s | 1s | -0.017329 | 0/6 |
| deribit-option-flow | deribit_option_trade_count_60s | 5s | -0.016905 | 0/6 |
| cross-exchange-book | binance_spot_l1_imbalance_mean_60s | 5s | -0.015431 | 0/6 |
| cross-exchange-book | deribit_perpetual_top5_imbalance | 60s | -0.014448 | 1/6 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_15s | 1s | -0.013855 | 1/6 |
| cross-exchange-book | binance_depth_pressure_mean_15s | 15s | -0.013786 | 1/6 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_15s | 60s | -0.012745 | 0/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 1s | -0.012530 | 1/6 |
| cross-exchange-book | binance_perpetual_l1_imbalance_mean_60s | 5s | -0.012368 | 1/6 |
| cross-exchange-book | coinbase_spot_l1_imbalance_mean_15s | 15s | -0.012237 | 1/6 |
| cross-exchange-book | coinbase_spot_l1_imbalance_mean_60s | 5s | -0.012206 | 1/6 |
| deribit-option-flow | deribit_option_trade_count_5s | 60s | -0.011723 | 1/6 |
| deribit-option-flow | deribit_option_trade_amount_5s | 60s | -0.011723 | 1/6 |
| deribit-option-flow | deribit_option_trade_count_15s | 15s | -0.010806 | 0/6 |
| deribit-option-flow | deribit_option_trade_amount_15s | 15s | -0.010806 | 0/6 |
| deribit-option-flow | deribit_option_trade_amount_60s | 5s | -0.010774 | 1/6 |

Machine-readable results: `data/benchmarks/live-fast-feature-early-screen.json`.
