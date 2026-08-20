# Live external-feature early screen — 2026-08-17

Generated at 2026-08-19T17:06:55.402Z after **29.733 hours of observed target coverage**. The latest collector session is 4 wall-clock seconds old; the archive spans 55.381 wall-clock hours including gaps.

Latest target observation: 2026-08-19T17:06:55.000Z; staleness: 0.4s.

**Current interpretation:** Early 1s-1m screen. Retain only as preliminary evidence; this window contains too few independent slow-horizon outcomes.

## What one hour and one day can establish

- **1 hour:** data-quality and alignment validation, plus a smoke test for unusually large 1s–15s effects. It is not a rejection test.
- **1 day:** an early held-out screen for 1s–1m targets. It still represents one market regime, so negative results are not enough to discard a varying feature.
- **7+ days:** first useful separated-day evidence for minute-cadence options/mempool and 15-minute GDELT measurements.
- **30–90 days:** needed for a credible 1h target test because a day contains only 24 non-overlapping 1h outcomes.

| target | independent outcomes in 1h | independent outcomes in 1d | earliest use | provisional history |
|---:|---:|---:|---|---:|
| 1s | 3600 | 86400 | 1h feed/large-effect smoke test; 1d early screen | 3-7d |
| 5s | 720 | 17280 | 1h feed/large-effect smoke test; 1d early screen | 3-7d |
| 15s | 240 | 5760 | 1h feed/large-effect smoke test; 1d early screen | 3-7d |
| 1m | 60 | 1440 | 1d early screen | 7-14d |
| 5m | 12 | 288 | 1d is weak; use at least 7d | 14-30d |
| 15m | 4 | 96 | 1d is weak; use at least 7d | 14-30d |
| 30m | 2 | 48 | 1d is not a predictive validation window | 30-90d |
| 1h | 1 | 24 | 1d is not a predictive validation window | 30-90d |

## Storage

- Stored across all trials: 407.64 MiB; active session: 0.05 MiB; projected active rate: 1061.20 MiB/day.
- Treat a projection from the first 15 minutes as an upper-biased startup estimate: one immediate options surface, GDELT pull, and mempool snapshot have not yet been amortized over their normal cadences.
- Unresolved slow candidate feeds in the active session: 0.04 MiB; projected 899.40 MiB/day.
- Evidence duration is observed target coverage, with gaps capped at five carried seconds; wall-clock age is never treated as data. Storage rate alone uses latest-session wall time because bytes accrue with elapsed time. Normal collection now stores causal 1s book summaries instead of full high-frequency books; --raw-books is diagnostic-only.

## Feed inventory

| source | files | stored | active-session stored | projected/day |
|---|---:|---:|---:|---:|
| kraken-btcusd-book | 2 | 87.98 MiB | 0.00 MiB | 0.00 MiB |
| coinbase-btcusd-level2 | 2 | 76.26 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-depth-diff | 2 | 58.32 MiB | 0.00 MiB | 0.00 MiB |
| cross-exchange-book-1s | 2 | 54.54 MiB | 0.00 MiB | 41.94 MiB |
| binance-spot-aggtrade | 4 | 41.18 MiB | 0.00 MiB | 59.01 MiB |
| deribit-btc-perpetual-book | 2 | 28.65 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-book-ticker | 2 | 19.56 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-mark-price | 2 | 14.15 MiB | 0.00 MiB | 12.64 MiB |
| deribit-btc-perpetual-trades | 2 | 6.46 MiB | 0.00 MiB | 18.11 MiB |
| mempool-live | 4 | 5.45 MiB | 0.00 MiB | 67.15 MiB |
| binance-usdm-liquidations | 2 | 4.51 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-option-surface-raw | 129 | 4.06 MiB | 0.04 MiB | 777.93 MiB |
| deribit-btc-option-summary | 4 | 2.82 MiB | 0.00 MiB | 32.91 MiB |
| deribit-btc-option-trades | 2 | 2.56 MiB | 0.00 MiB | 7.16 MiB |
| binance-spot-depth-snapshot | 2 | 0.66 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-premium-index | 4 | 0.35 MiB | 0.00 MiB | 4.10 MiB |
| gdelt-crypto-news | 4 | 0.12 MiB | 0.00 MiB | 17.31 MiB |
| binance-usdm-book-ticker-session | 4 | 0.01 MiB | 0.00 MiB | 2.61 MiB |
| kraken-btcusd-book-session | 4 | 0.00 MiB | 0.00 MiB | 2.08 MiB |
| deribit-btc-perpetual-book-session | 4 | 0.00 MiB | 0.00 MiB | 2.20 MiB |
| binance-spot-depth-diff-session | 4 | 0.00 MiB | 0.00 MiB | 2.55 MiB |
| collector-session | 4 | 0.00 MiB | 0.00 MiB | 3.47 MiB |
| coinbase-btcusd-level2-session | 4 | 0.00 MiB | 0.00 MiB | 2.32 MiB |
| binance-usdm-mark-price-session | 4 | 0.00 MiB | 0.00 MiB | 2.63 MiB |
| binance-spot-aggtrade-session | 4 | 0.00 MiB | 0.00 MiB | 2.53 MiB |
| binance-usdm-liquidations-session | 4 | 0.00 MiB | 0.00 MiB | 2.55 MiB |

## Candidate feature health

| family | feature | observations | changes | distinct | health |
|---|---|---:|---:|---:|---|
| futures-premium | premium_bps | 1792 | 1790 | 1791 | varying |
| futures-premium | funding_rate_bps | 1792 | 1176 | 1076 | varying |
| futures-premium | mark_index_abs_gap_bps | 1792 | 1790 | 1791 | varying |
| options | atm_iv_1d | 1792 | 1529 | 688 | varying |
| options | atm_iv_7d | 1792 | 1462 | 455 | varying |
| options | atm_iv_30d | 1792 | 944 | 171 | varying |
| options | atm_iv_term_7d_minus_1d | 1792 | 1645 | 757 | varying |
| options | atm_iv_term_30d_minus_7d | 1792 | 1578 | 453 | varying |
| options | put_call_25d_skew_1d | 1792 | 1675 | 544 | varying |
| options | put_call_25d_skew_7d | 1792 | 1584 | 387 | varying |
| options | put_call_25d_skew_30d | 1792 | 1386 | 230 | varying |
| options | total_call_put_oi_imbalance | 1792 | 1274 | 1266 | varying |
| options | one_day_call_put_oi_imbalance | 1792 | 505 | 505 | varying |
| options | log_distance_to_major_strike | 1792 | 1790 | 1778 | varying |
| options | hours_to_next_expiry | 1792 | 1791 | 1792 | varying |
| mempool | transaction_count | 1792 | 1791 | 1647 | varying |
| mempool | virtual_size | 1792 | 1791 | 1792 | varying |
| mempool | total_fee_btc | 1792 | 1791 | 1791 | varying |
| mempool | mean_fee_sat_vbyte | 1792 | 1791 | 1792 | varying |
| mempool | fastest_fee | 1792 | 258 | 6 | varying |
| mempool | half_hour_fee | 1792 | 121 | 5 | varying |
| mempool | hour_fee | 1792 | 24 | 4 | varying |
| mempool | economy_fee | 1792 | 14 | 2 | near-constant |
| mempool | minimum_fee | 1792 | 0 | 1 | near-constant |
| mempool | projected_first_block_vsize | 1792 | 1788 | 409 | varying |
| mempool | projected_first_block_fee_range_high | 1792 | 446 | 331 | varying |
| news-gdelt | crypto_terms_per_million | 125 | 120 | 121 | varying |
| news-gdelt | story_count | 125 | 106 | 18 | varying |
| news-gdelt | source_count | 125 | 106 | 12 | varying |
| news-gdelt | mean_tone | 125 | 120 | 120 | varying |
| news-gdelt | mean_positive | 125 | 120 | 120 | varying |
| news-gdelt | mean_negative | 125 | 120 | 120 | varying |
| news-gdelt | mean_polarity | 125 | 120 | 120 | varying |

## Predictive smoke/early screen

Scores are out-of-sample gain over trailing-return and trailing-volatility bins. Positive bits/target is better. The first 60% of this live window fits all quantile edges and categorical probabilities; the last 40% is evaluated in six chronological blocks.
This is a single chronological split. During the smoke stage, a few appended observations can move the split boundary and materially reorder sparse categorical scores; do not treat the ranking as a selected model basis until it survives separated-day evaluation.

| family | feature | target | eval rows | effective outcomes | bits/target | positive blocks | decision | evidence |
|---|---|---:|---:|---:|---:|---:|---|---|
| options | atm_iv_1d | 1h | 259 | 46 | 0.742111 | 5/6 | promising | early |
| options | hours_to_next_expiry | 1h | 259 | 46 | 0.468322 | 4/6 | inconclusive | early |
| options | atm_iv_term_7d_minus_1d | 1h | 259 | 46 | 0.347425 | 6/6 | promising | early |
| options | atm_iv_7d | 1h | 259 | 46 | 0.214755 | 4/6 | inconclusive | early |
| news-gdelt | mean_tone | 15m | 30 | 30 | 0.194989 | 3/3 | promising | early |
| news-gdelt | mean_polarity | 5m | 40 | 40 | 0.191551 | 4/5 | promising | early |
| news-gdelt | mean_polarity | 1m | 46 | 46 | 0.181382 | 3/5 | inconclusive | early |
| options | atm_iv_term_30d_minus_7d | 1h | 259 | 46 | 0.175615 | 4/6 | inconclusive | early |
| news-gdelt | mean_negative | 15m | 30 | 30 | 0.147787 | 2/3 | inconclusive | early |
| news-gdelt | source_count | 1s | 50 | 50 | 0.080297 | 4/6 | inconclusive | early |
| options | total_call_put_oi_imbalance | 1h | 259 | 46 | 0.066482 | 5/6 | promising | early |
| news-gdelt | crypto_terms_per_million | 5m | 40 | 40 | 0.064287 | 2/5 | inconclusive | early |
| news-gdelt | mean_tone | 1s | 50 | 50 | 0.063786 | 3/6 | inconclusive | early |
| mempool | projected_first_block_fee_range_high | 1h | 259 | 46 | 0.034507 | 4/6 | inconclusive | early |
| mempool | projected_first_block_vsize | 1h | 259 | 46 | 0.032628 | 4/6 | inconclusive | early |
| news-gdelt | mean_negative | 1s | 50 | 50 | 0.029834 | 3/6 | inconclusive | early |
| options | put_call_25d_skew_1d | 1h | 259 | 46 | 0.028990 | 3/6 | inconclusive | early |
| news-gdelt | mean_positive | 5m | 40 | 40 | 0.027489 | 2/5 | inconclusive | early |
| options | put_call_25d_skew_1d | 5s | 712 | 712 | 0.026861 | 4/6 | inconclusive | early |
| news-gdelt | story_count | 30m | 23 | 23 | 0.025404 | 1/2 | inconclusive | early |
| news-gdelt | crypto_terms_per_million | 1m | 46 | 46 | 0.017430 | 3/5 | inconclusive | early |
| options | total_call_put_oi_imbalance | 5s | 712 | 712 | 0.014601 | 3/6 | inconclusive | early |
| news-gdelt | source_count | 15m | 30 | 30 | 0.010084 | 2/3 | inconclusive | early |
| futures-premium | funding_rate_bps | 1m | 681 | 681 | 0.009799 | 2/6 | inconclusive | early |
| futures-premium | mark_index_abs_gap_bps | 15m | 447 | 61 | 0.005803 | 3/6 | inconclusive | early |
| futures-premium | premium_bps | 15m | 447 | 61 | 0.004693 | 3/6 | inconclusive | early |
| mempool | minimum_fee | 1s | 714 | 714 | 0.000000 | 0/6 | weak-in-this-window | early |
| mempool | minimum_fee | 5s | 712 | 712 | 0.000000 | 0/6 | weak-in-this-window | early |
| mempool | minimum_fee | 15s | 704 | 704 | 0.000000 | 0/6 | weak-in-this-window | early |
| mempool | minimum_fee | 1m | 681 | 681 | 0.000000 | 0/6 | weak-in-this-window | early |

## Retention decision

1. Stop a feed immediately only if it is malformed, timestamp-misaligned, stale, or near-constant.
2. At 24h, promote repeated positive results to `early`; do not reject a varying feature from a single day.
3. Starting with separated days, stop a feature family only when its upper uncertainty bound is below the chosen material gain threshold across target horizons.
4. Compact high-volume raw books into causal 1s derived features after their reconstruction tests; the unresolved slow feeds themselves are cheap enough to retain through the required evidence window.

Machine-readable results are in `data/benchmarks/live-external-early-screen.json`.
