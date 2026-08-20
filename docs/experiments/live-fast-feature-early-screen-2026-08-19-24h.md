# Live fast-feature early screen — 2026-08-19

Generated at 2026-08-19T17:07:58.011Z. Target coverage is **29.736 observed hours**.

## Interpretation

- Degraded compact books: krakenSpot (69.9% valid). Treat their features as missing behind observed/age masks.
- 30 feature/horizon pairs clear the strict early-effect rule; they are candidates for the 3-7 day confirmation, not production promotion.
- 44 feature/horizon pairs are consistently negative in this window, but one day is not enough for permanent rejection.
- Repeat after 3 and 7 observed days. Promote only effects that retain positive held-out gain across separated days and in a joint ablation against the established basis.
- This is explicitly an early effect and broken-feed screen. It does not promote or permanently reject model inputs.

## Causal validation

- Every fast feature bucket from second t first becomes available at t+1; targets begin from the completed price at t.
- BTCUSDT end-of-completed-second log return at 1s, 5s, 15s, and 60s, aligned by local receive time
- Categorical conditional density from prior same-horizon return and trailing absolute-return state.
- Baseline plus one feature discretized into training-only tertiles; target quartiles are also training-only.
- First 60% trains; final 40% evaluates in six chronological blocks. Three deterministic feature permutations estimate finite-sample/extra-state bias.
- One live day is an early effect/broken-feed screen, not promotion or rejection evidence. Scores are exploratory and not multiple-testing adjusted.

## Feed health

Compact book rows: 72,391; observed coverage: 20.190h; maximum gap: 14s.

| venue | valid rows | valid fraction | median age | p99 age | status |
|---|---:|---:|---:|---:|---|
| binanceSpot | 72,230 | 99.78% | 61.0ms | 213.7ms | healthy |
| coinbaseSpot | 72,385 | 99.99% | 35.0ms | 105.0ms | healthy |
| krakenSpot | 50,594 | 69.89% | 59.0ms | 749.1ms | degraded |
| deribitPerpetual | 72,380 | 99.98% | 171.0ms | 409.0ms | healthy |
| binancePerpetual | 72,296 | 99.87% | 38.0ms | 463.0ms | healthy |

Kraken-derived scores from this archive are marked `feed-contaminated` and excluded from all rankings. The collector failed to truncate the reconstructed book to the subscribed depth; [Kraken's reconstruction rules](https://docs.kraken.com/exchange/guides/websockets/book-checksum-v2) explicitly say that zero-quantity removals are not sent for levels that merely fall out of scope. The collector now truncates to depth 100 and reconnects for a fresh snapshot whenever the reconstructed book crosses. Historical compact rows are not rewritten.

All-market liquidation messages: 35,945; BTCUSDT messages: 1,236 (3.44%). Absence is encoded as zero rather than dropping non-event seconds.

Deribit perpetual trades: 91,657; option trades: 13,341.

## Family summary

| family | evaluated pairs | strict early effects | weak in window | best clean feature | target | bits/target | lower block bound |
|---|---:|---:|---:|---|---:|---:|---:|
| cross-exchange-book | 160 | 12 | 24 | binance_perpetual_basis_bps_mean_60s | 15s | 0.058067 | -0.025139 |
| btc-liquidations | 48 | 0 | 4 | btc_liquidation_count_60s | 15s | 0.023446 | -0.009898 |
| deribit-perpetual-flow | 48 | 18 | 0 | deribit_perpetual_trade_count_60s | 60s | 0.054064 | -0.000535 |
| deribit-option-flow | 48 | 0 | 16 | deribit_option_trade_count_5s | 1s | 0.016540 | -0.005325 |

The clean strict candidates concentrate in Deribit perpetual **activity** (trade count/amount), Binance displayed-depth **churn**, and a smaller set of rolling book/cross-venue states. BTC liquidation flow and Deribit option-trade flow have no strict early winner. This pattern currently supports magnitude/activity-state information more than stable directional information.

## Strict early-effect candidates

A row must have a positive 95% block bound, at least 5/6 positive chronological blocks, and exceed its shuffled-feature control. These remain confirmation candidates only.

| family | feature | target | bits/target | lower block bound | shuffled control | positive blocks | effective outcomes |
|---|---|---:|---:|---:|---:|---:|---:|
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 60s | 0.049764 | 0.004377 | -0.002616 | 5/6 | 474 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 15s | 0.044612 | 0.007377 | -0.005438 | 6/6 | 1,926 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 15s | 0.040338 | 0.000556 | -0.002493 | 5/6 | 1,926 |
| cross-exchange-book | binance_depth_churn_log_quote | 60s | 0.033191 | 0.002516 | -0.001961 | 6/6 | 474 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 15s | 0.030953 | 0.003177 | 0.000124 | 6/6 | 1,926 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 15s | 0.030554 | 0.002380 | -0.002567 | 6/6 | 1,926 |
| cross-exchange-book | binance_depth_churn_log_quote | 15s | 0.026555 | 0.004240 | 0.000270 | 6/6 | 1,925 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 60s | 0.026340 | 0.002595 | -0.002436 | 6/6 | 474 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 5s | 0.025369 | 0.010304 | 0.000260 | 6/6 | 5,805 |
| cross-exchange-book | binance_depth_churn_log_quote | 5s | 0.023350 | 0.006724 | 0.002054 | 6/6 | 5,804 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 15s | 0.023249 | 0.002152 | 0.002344 | 5/6 | 1,926 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 60s | 0.022372 | 0.003288 | -0.003020 | 6/6 | 474 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 5s | 0.021905 | 0.004516 | -0.000852 | 5/6 | 5,805 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_1s | 1s | 0.019575 | 0.000081 | -0.002144 | 6/6 | 29,000 |
| cross-exchange-book | binance_spot_l1_imbalance_mean_60s | 15s | 0.016886 | 0.008518 | -0.000054 | 6/6 | 1,930 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_15s | 15s | 0.013339 | 0.001619 | -0.001709 | 5/6 | 1,925 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_5s | 1s | 0.012118 | 0.000757 | -0.001463 | 6/6 | 29,000 |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 5s | 0.011634 | 0.005668 | -0.000867 | 6/6 | 5,805 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_5s | 60s | 0.011417 | 0.003085 | -0.001626 | 6/6 | 474 |
| cross-exchange-book | coinbase_spot_l1_imbalance_mean_5s | 60s | 0.011182 | 0.003905 | -0.001242 | 6/6 | 474 |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 5s | 0.010912 | 0.005314 | -0.000843 | 6/6 | 5,805 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_5s | 15s | 0.010858 | 0.003184 | 0.000325 | 6/6 | 1,926 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_1s | 15s | 0.010295 | 0.003405 | -0.001343 | 6/6 | 1,926 |
| cross-exchange-book | best_executable_spread_bps | 1s | 0.005825 | 0.000636 | -0.001194 | 6/6 | 28,891 |
| cross-exchange-book | spot_mid_dispersion_bps | 1s | 0.005815 | 0.000662 | -0.001121 | 6/6 | 28,891 |
| cross-exchange-book | binance_spot_l1_imbalance_mean_60s | 1s | 0.004466 | 0.000086 | 0.000264 | 5/6 | 28,700 |
| cross-exchange-book | binance_remove_imbalance | 5s | 0.003710 | 0.000391 | 0.000870 | 5/6 | 5,804 |
| cross-exchange-book | spot_mid_dispersion_bps_mean_5s | 1s | 0.003483 | 0.001758 | -0.000399 | 6/6 | 28,886 |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_60s | 1s | 0.002604 | 0.000193 | -0.001055 | 5/6 | 29,000 |
| cross-exchange-book | binance_add_imbalance | 15s | 0.002446 | 0.000806 | -0.000596 | 6/6 | 1,925 |

## Best exploratory scores

| family | feature | target | bits/target | excess over shuffle | blocks | classification |
|---|---|---:|---:|---:|---:|---|
| cross-exchange-book | binance_perpetual_basis_bps_mean_60s | 15s | 0.058067 | 0.058438 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 60s | 0.054064 | 0.053338 | 5/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 60s | 0.049764 | 0.052379 | 5/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_60s | 15s | 0.044612 | 0.050050 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 15s | 0.040338 | 0.042831 | 5/6 | large-early-effect |
| cross-exchange-book | binance_depth_churn_log_quote | 60s | 0.033191 | 0.035152 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps_mean_15s | 15s | 0.033017 | 0.030821 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 15s | 0.030953 | 0.030828 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 15s | 0.030554 | 0.033121 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps_mean_5s | 15s | 0.028963 | 0.026198 | 4/6 | inconclusive |
| cross-exchange-book | binance_depth_churn_log_quote | 15s | 0.026555 | 0.026285 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 60s | 0.026340 | 0.028776 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_5s | 1s | 0.026024 | 0.025769 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 5s | 0.025369 | 0.025109 | 6/6 | large-early-effect |
| cross-exchange-book | binance_depth_churn_log_quote | 1s | 0.025177 | 0.025457 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 1s | 0.024666 | 0.025029 | 6/6 | inconclusive |
| btc-liquidations | btc_liquidation_count_60s | 15s | 0.023446 | 0.024798 | 4/6 | inconclusive |
| btc-liquidations | btc_liquidation_amount_60s | 15s | 0.023446 | 0.024798 | 4/6 | inconclusive |
| cross-exchange-book | binance_depth_churn_log_quote | 5s | 0.023350 | 0.021297 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 15s | 0.023249 | 0.020905 | 5/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_1s | 1s | 0.022835 | 0.023580 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_1s | 1s | 0.022835 | 0.023580 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_60s | 15s | 0.022741 | 0.026304 | 5/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_60s | 60s | 0.022683 | 0.023311 | 5/6 | inconclusive |
| cross-exchange-book | best_executable_spread_bps | 15s | 0.022563 | 0.021258 | 3/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_5s | 60s | 0.022372 | 0.025392 | 6/6 | large-early-effect |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 1s | 0.022121 | 0.022840 | 5/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_15s | 5s | 0.021905 | 0.022757 | 5/6 | large-early-effect |
| cross-exchange-book | spot_mid_dispersion_bps | 15s | 0.021310 | 0.020047 | 3/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_15s | 60s | 0.021243 | 0.022183 | 5/6 | inconclusive |
| cross-exchange-book | binance_perpetual_basis_bps_mean_60s | 60s | 0.020136 | 0.021267 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_count_1s | 15s | 0.019768 | 0.020123 | 6/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_amount_1s | 15s | 0.019768 | 0.020123 | 6/6 | inconclusive |
| cross-exchange-book | spot_mid_dispersion_bps_mean_5s | 15s | 0.019676 | 0.019642 | 3/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_1s | 1s | 0.019575 | 0.021719 | 6/6 | large-early-effect |
| cross-exchange-book | binance_perpetual_basis_bps | 15s | 0.017575 | 0.013369 | 4/6 | inconclusive |
| deribit-perpetual-flow | deribit_perpetual_trade_imbalance_60s | 60s | 0.017353 | 0.019859 | 5/6 | inconclusive |
| cross-exchange-book | binance_spot_l1_imbalance_mean_60s | 15s | 0.016886 | 0.016940 | 6/6 | large-early-effect |
| deribit-option-flow | deribit_option_trade_count_5s | 1s | 0.016540 | 0.017607 | 5/6 | inconclusive |
| deribit-option-flow | deribit_option_trade_amount_5s | 1s | 0.016540 | 0.017607 | 5/6 | inconclusive |

## Consistently negative rows in this window

These are candidates for later rejection, not rejected features. A single live day cannot distinguish a genuinely useless input from a regime-specific failure.

| family | feature | target | bits/target | blocks |
|---|---|---:|---:|---:|
| btc-liquidations | btc_liquidation_imbalance_60s | 60s | -0.084363 | 0/6 |
| cross-exchange-book | binance_spot_spread_bps | 60s | -0.055860 | 0/6 |
| cross-exchange-book | deribit_perpetual_spread_bps | 60s | -0.055287 | 0/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 60s | -0.052266 | 1/6 |
| cross-exchange-book | binance_perpetual_spread_bps | 60s | -0.051569 | 0/6 |
| btc-liquidations | btc_liquidation_imbalance_15s | 60s | -0.050008 | 0/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 15s | -0.049621 | 0/6 |
| btc-liquidations | btc_liquidation_imbalance_15s | 15s | -0.043232 | 0/6 |
| cross-exchange-book | coinbase_spot_spread_bps | 60s | -0.040609 | 0/6 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_60s | 60s | -0.040104 | 1/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 5s | -0.028377 | 1/6 |
| cross-exchange-book | binance_perpetual_basis_bps | 1s | -0.023631 | 0/6 |
| cross-exchange-book | binance_perpetual_basis_bps | 60s | -0.023282 | 1/6 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_15s | 60s | -0.022737 | 1/6 |
| cross-exchange-book | binance_depth_pressure_mean_60s | 60s | -0.020387 | 1/6 |
| cross-exchange-book | deribit_perpetual_top5_imbalance | 60s | -0.018845 | 1/6 |
| cross-exchange-book | binance_perpetual_basis_bps_mean_5s | 1s | -0.016479 | 0/6 |
| cross-exchange-book | deribit_perpetual_l1_imbalance_mean_5s | 60s | -0.015030 | 1/6 |
| btc-liquidations | btc_liquidation_imbalance_60s | 15s | -0.014383 | 0/6 |
| deribit-option-flow | deribit_option_trade_count_60s | 5s | -0.013522 | 0/6 |
| cross-exchange-book | binance_depth_pressure_mean_15s | 15s | -0.013355 | 0/6 |
| deribit-option-flow | deribit_option_trade_count_15s | 15s | -0.012235 | 0/6 |
| deribit-option-flow | deribit_option_trade_amount_15s | 15s | -0.012235 | 0/6 |
| cross-exchange-book | binance_spot_l1_imbalance_mean_60s | 5s | -0.009284 | 1/6 |
| deribit-option-flow | deribit_option_trade_amount_60s | 15s | -0.009080 | 1/6 |
| deribit-option-flow | deribit_option_trade_imbalance_60s | 1s | -0.008959 | 0/6 |
| deribit-option-flow | deribit_option_trade_count_5s | 60s | -0.008155 | 1/6 |
| deribit-option-flow | deribit_option_trade_amount_5s | 60s | -0.008155 | 1/6 |
| deribit-option-flow | deribit_option_trade_amount_60s | 5s | -0.007368 | 1/6 |
| cross-exchange-book | binance_perpetual_basis_bps_mean_15s | 1s | -0.006541 | 0/6 |

Machine-readable results: `data/benchmarks/live-fast-feature-early-screen.json`.
