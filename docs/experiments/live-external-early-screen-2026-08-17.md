# Live external-feature early screen — 2026-08-17

Generated at 2026-08-20T13:03:18.470Z after **49.672 hours of observed target coverage**. The latest collector session is 71787 wall-clock seconds old; the archive spans 75.322 wall-clock hours including gaps.

Latest target observation: 2026-08-20T13:03:23.000Z; staleness: 0.0s.

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

- Stored across all trials: 549.56 MiB; active session: 141.97 MiB; projected active rate: 170.87 MiB/day.
- Treat a projection from the first 15 minutes as an upper-biased startup estimate: one immediate options surface, GDELT pull, and mempool snapshot have not yet been amortized over their normal cadences.
- Unresolved slow candidate feeds in the active session: 8.90 MiB; projected 10.72 MiB/day.
- Evidence duration is observed target coverage, with gaps capped at five carried seconds; wall-clock age is never treated as data. Storage rate alone uses latest-session wall time because bytes accrue with elapsed time. Normal collection now stores causal 1s book summaries instead of full high-frequency books; --raw-books is diagnostic-only.

## Feed inventory

| source | files | stored | active-session stored | projected/day |
|---|---:|---:|---:|---:|
| cross-exchange-book-1s | 3 | 108.62 MiB | 54.08 MiB | 65.09 MiB |
| kraken-btcusd-book | 2 | 87.98 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-aggtrade | 5 | 86.94 MiB | 45.76 MiB | 55.07 MiB |
| coinbase-btcusd-level2 | 2 | 76.26 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-depth-diff | 2 | 58.32 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-perpetual-book | 2 | 28.65 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-mark-price | 3 | 28.12 MiB | 13.98 MiB | 16.83 MiB |
| binance-usdm-book-ticker | 2 | 19.56 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-perpetual-trades | 3 | 17.00 MiB | 10.54 MiB | 12.68 MiB |
| binance-usdm-liquidations | 3 | 9.36 MiB | 4.85 MiB | 5.84 MiB |
| mempool-live | 5 | 9.06 MiB | 3.62 MiB | 4.36 MiB |
| deribit-btc-option-surface-raw | 208 | 7.13 MiB | 3.11 MiB | 3.74 MiB |
| deribit-btc-option-trades | 3 | 6.41 MiB | 3.85 MiB | 4.64 MiB |
| deribit-btc-option-summary | 5 | 4.69 MiB | 1.87 MiB | 2.26 MiB |
| binance-spot-depth-snapshot | 2 | 0.66 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-premium-index | 5 | 0.57 MiB | 0.23 MiB | 0.27 MiB |
| gdelt-crypto-news | 5 | 0.19 MiB | 0.08 MiB | 0.09 MiB |
| binance-usdm-book-ticker-session | 5 | 0.01 MiB | 0.00 MiB | 0.00 MiB |
| kraken-btcusd-book-session | 5 | 0.01 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-perpetual-book-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| coinbase-btcusd-level2-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-depth-diff-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| collector-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-mark-price-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-aggtrade-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-liquidations-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |

## Candidate feature health

| family | feature | observations | changes | distinct | health |
|---|---|---:|---:|---:|---|
| futures-premium | premium_bps | 2988 | 2986 | 2987 | varying |
| futures-premium | funding_rate_bps | 2988 | 1335 | 1218 | varying |
| futures-premium | mark_index_abs_gap_bps | 2988 | 2986 | 2987 | varying |
| options | atm_iv_1d | 2988 | 2657 | 1434 | varying |
| options | atm_iv_7d | 2988 | 2450 | 982 | varying |
| options | atm_iv_30d | 2988 | 1751 | 472 | varying |
| options | atm_iv_term_7d_minus_1d | 2988 | 2792 | 1381 | varying |
| options | atm_iv_term_30d_minus_7d | 2988 | 2615 | 904 | varying |
| options | put_call_25d_skew_1d | 2988 | 2814 | 955 | varying |
| options | put_call_25d_skew_7d | 2988 | 2692 | 679 | varying |
| options | put_call_25d_skew_30d | 2988 | 2402 | 379 | varying |
| options | total_call_put_oi_imbalance | 2988 | 2399 | 2389 | varying |
| options | one_day_call_put_oi_imbalance | 2988 | 1039 | 1035 | varying |
| options | log_distance_to_major_strike | 2988 | 2986 | 2965 | varying |
| options | hours_to_next_expiry | 2988 | 2987 | 2988 | varying |
| mempool | transaction_count | 2988 | 2987 | 2760 | varying |
| mempool | virtual_size | 2988 | 2987 | 2988 | varying |
| mempool | total_fee_btc | 2988 | 2987 | 2987 | varying |
| mempool | mean_fee_sat_vbyte | 2988 | 2987 | 2988 | varying |
| mempool | fastest_fee | 2988 | 439 | 6 | varying |
| mempool | half_hour_fee | 2988 | 210 | 5 | varying |
| mempool | hour_fee | 2988 | 50 | 4 | varying |
| mempool | economy_fee | 2988 | 32 | 2 | near-constant |
| mempool | minimum_fee | 2988 | 0 | 1 | near-constant |
| mempool | projected_first_block_vsize | 2988 | 2981 | 431 | varying |
| mempool | projected_first_block_fee_range_high | 2988 | 747 | 552 | varying |
| news-gdelt | crypto_terms_per_million | 204 | 199 | 200 | varying |
| news-gdelt | story_count | 204 | 172 | 20 | varying |
| news-gdelt | source_count | 204 | 167 | 13 | varying |
| news-gdelt | mean_tone | 204 | 198 | 197 | varying |
| news-gdelt | mean_positive | 204 | 198 | 197 | varying |
| news-gdelt | mean_negative | 204 | 198 | 196 | varying |
| news-gdelt | mean_polarity | 204 | 198 | 197 | varying |

## Predictive smoke/early screen

Scores are out-of-sample gain over trailing-return and trailing-volatility bins. Positive bits/target is better. The first 60% of this live window fits all quantile edges and categorical probabilities; the last 40% is evaluated in six chronological blocks.
This is a single chronological split. During the smoke stage, a few appended observations can move the split boundary and materially reorder sparse categorical scores; do not treat the ranking as a selected model basis until it survives separated-day evaluation.

| family | feature | target | eval rows | effective outcomes | bits/target | positive blocks | decision | evidence |
|---|---|---:|---:|---:|---:|---:|---|---|
| options | atm_iv_30d | 30m | 792 | 26 | 0.114613 | 5/6 | promising | early |
| news-gdelt | mean_tone | 15m | 62 | 61 | 0.078755 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_7d | 15s | 1182 | 1182 | 0.043902 | 6/6 | promising | early |
| options | put_call_25d_skew_7d | 15m | 914 | 60 | 0.043444 | 4/6 | inconclusive | early |
| futures-premium | funding_rate_bps | 15s | 1183 | 1183 | 0.040187 | 6/6 | promising | early |
| futures-premium | funding_rate_bps | 30m | 792 | 26 | 0.033925 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_30d | 15s | 1182 | 1182 | 0.024686 | 6/6 | promising | early |
| options | hours_to_next_expiry | 15m | 914 | 60 | 0.019143 | 3/6 | inconclusive | early |
| news-gdelt | mean_negative | 15m | 62 | 61 | 0.018890 | 2/6 | inconclusive | early |
| mempool | half_hour_fee | 15m | 914 | 60 | 0.018715 | 4/6 | inconclusive | early |
| options | atm_iv_term_7d_minus_1d | 15s | 1182 | 1182 | 0.018212 | 6/6 | promising | early |
| news-gdelt | mean_polarity | 5s | 80 | 80 | 0.016613 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_7d | 1m | 1158 | 1157 | 0.016046 | 6/6 | promising | early |
| options | atm_iv_term_30d_minus_7d | 15m | 914 | 60 | 0.015125 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_30d | 1m | 1158 | 1157 | 0.013820 | 6/6 | promising | early |
| futures-premium | funding_rate_bps | 1m | 1158 | 1157 | 0.011165 | 5/6 | promising | early |
| mempool | projected_first_block_fee_range_high | 5s | 1191 | 1191 | 0.007612 | 4/6 | inconclusive | early |
| options | atm_iv_1d | 5m | 1059 | 211 | 0.007211 | 4/6 | inconclusive | early |
| news-gdelt | mean_polarity | 5m | 72 | 72 | 0.005887 | 4/6 | inconclusive | early |
| options | atm_iv_term_7d_minus_1d | 1m | 1158 | 1157 | 0.004809 | 4/6 | inconclusive | early |
| options | atm_iv_7d | 15m | 914 | 60 | 0.003858 | 3/6 | inconclusive | early |
| options | put_call_25d_skew_7d | 5m | 1059 | 211 | 0.003719 | 2/6 | inconclusive | early |
| options | atm_iv_1d | 15s | 1182 | 1182 | 0.002584 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_30d | 5m | 1059 | 211 | 0.001988 | 2/6 | inconclusive | early |
| options | put_call_25d_skew_1d | 5m | 1059 | 211 | 0.001988 | 3/6 | inconclusive | early |
| options | put_call_25d_skew_7d | 1s | 1192 | 1192 | 0.001793 | 3/6 | inconclusive | early |
| mempool | minimum_fee | 1s | 1192 | 1192 | 0.000000 | 0/6 | weak-in-this-window | early |
| mempool | minimum_fee | 5s | 1191 | 1191 | 0.000000 | 0/6 | weak-in-this-window | early |
| mempool | minimum_fee | 15s | 1183 | 1183 | 0.000000 | 0/6 | weak-in-this-window | early |
| mempool | minimum_fee | 1m | 1159 | 1158 | 0.000000 | 0/6 | weak-in-this-window | early |

## Retention decision

1. Stop a feed immediately only if it is malformed, timestamp-misaligned, stale, or near-constant.
2. At 24h, promote repeated positive results to `early`; do not reject a varying feature from a single day.
3. Starting with separated days, stop a feature family only when its upper uncertainty bound is below the chosen material gain threshold across target horizons.
4. Compact high-volume raw books into causal 1s derived features after their reconstruction tests; the unresolved slow feeds themselves are cheap enough to retain through the required evidence window.

Machine-readable results are in `data/benchmarks/live-external-early-screen.json`.
