# Live external-feature early screen — 2026-08-17

Generated at 2026-08-20T07:44:17.930Z after **44.357 hours of observed target coverage**. The latest collector session is 52647 wall-clock seconds old; the archive spans 70.005 wall-clock hours including gaps.

Latest target observation: 2026-08-20T07:44:21.000Z; staleness: 0.0s.

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

- Stored across all trials: 505.19 MiB; active session: 97.60 MiB; projected active rate: 160.18 MiB/day.
- Treat a projection from the first 15 minutes as an upper-biased startup estimate: one immediate options surface, GDELT pull, and mempool snapshot have not yet been amortized over their normal cadences.
- Unresolved slow candidate feeds in the active session: 6.48 MiB; projected 10.64 MiB/day.
- Evidence duration is observed target coverage, with gaps capped at five carried seconds; wall-clock age is never treated as data. Storage rate alone uses latest-session wall time because bytes accrue with elapsed time. Normal collection now stores causal 1s book summaries instead of full high-frequency books; --raw-books is diagnostic-only.

## Feed inventory

| source | files | stored | active-session stored | projected/day |
|---|---:|---:|---:|---:|
| cross-exchange-book-1s | 3 | 94.21 MiB | 39.68 MiB | 65.11 MiB |
| kraken-btcusd-book | 2 | 87.98 MiB | 0.00 MiB | 0.00 MiB |
| coinbase-btcusd-level2 | 2 | 76.26 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-aggtrade | 5 | 70.23 MiB | 29.05 MiB | 47.67 MiB |
| binance-spot-depth-diff | 2 | 58.32 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-perpetual-book | 2 | 28.65 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-mark-price | 3 | 24.41 MiB | 10.26 MiB | 16.84 MiB |
| binance-usdm-book-ticker | 2 | 19.56 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-perpetual-trades | 3 | 13.12 MiB | 6.66 MiB | 10.93 MiB |
| mempool-live | 5 | 8.10 MiB | 2.66 MiB | 4.37 MiB |
| binance-usdm-liquidations | 3 | 7.75 MiB | 3.24 MiB | 5.32 MiB |
| deribit-btc-option-surface-raw | 187 | 6.26 MiB | 2.24 MiB | 3.68 MiB |
| deribit-btc-option-trades | 3 | 4.79 MiB | 2.23 MiB | 3.66 MiB |
| deribit-btc-option-summary | 5 | 4.17 MiB | 1.36 MiB | 2.23 MiB |
| binance-spot-depth-snapshot | 2 | 0.66 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-premium-index | 5 | 0.51 MiB | 0.17 MiB | 0.27 MiB |
| gdelt-crypto-news | 5 | 0.17 MiB | 0.05 MiB | 0.08 MiB |
| binance-usdm-book-ticker-session | 4 | 0.01 MiB | 0.00 MiB | 0.00 MiB |
| kraken-btcusd-book-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| deribit-btc-perpetual-book-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-depth-diff-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| coinbase-btcusd-level2-session | 5 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| collector-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-mark-price-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-spot-aggtrade-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |
| binance-usdm-liquidations-session | 4 | 0.00 MiB | 0.00 MiB | 0.00 MiB |

## Candidate feature health

| family | feature | observations | changes | distinct | health |
|---|---|---:|---:|---:|---|
| futures-premium | premium_bps | 2669 | 2667 | 2668 | varying |
| futures-premium | funding_rate_bps | 2669 | 1335 | 1218 | varying |
| futures-premium | mark_index_abs_gap_bps | 2669 | 2667 | 2668 | varying |
| options | atm_iv_1d | 2669 | 2351 | 1188 | varying |
| options | atm_iv_7d | 2669 | 2170 | 810 | varying |
| options | atm_iv_30d | 2669 | 1516 | 405 | varying |
| options | atm_iv_term_7d_minus_1d | 2669 | 2483 | 1153 | varying |
| options | atm_iv_term_30d_minus_7d | 2669 | 2316 | 748 | varying |
| options | put_call_25d_skew_1d | 2669 | 2501 | 785 | varying |
| options | put_call_25d_skew_7d | 2669 | 2387 | 548 | varying |
| options | put_call_25d_skew_30d | 2669 | 2127 | 305 | varying |
| options | total_call_put_oi_imbalance | 2669 | 2082 | 2073 | varying |
| options | one_day_call_put_oi_imbalance | 2669 | 805 | 804 | varying |
| options | log_distance_to_major_strike | 2669 | 2667 | 2646 | varying |
| options | hours_to_next_expiry | 2669 | 2668 | 2669 | varying |
| mempool | transaction_count | 2669 | 2668 | 2482 | varying |
| mempool | virtual_size | 2669 | 2668 | 2669 | varying |
| mempool | total_fee_btc | 2669 | 2668 | 2668 | varying |
| mempool | mean_fee_sat_vbyte | 2669 | 2668 | 2669 | varying |
| mempool | fastest_fee | 2669 | 389 | 6 | varying |
| mempool | half_hour_fee | 2669 | 179 | 5 | varying |
| mempool | hour_fee | 2669 | 40 | 4 | varying |
| mempool | economy_fee | 2669 | 24 | 2 | near-constant |
| mempool | minimum_fee | 2669 | 0 | 1 | near-constant |
| mempool | projected_first_block_vsize | 2669 | 2663 | 427 | varying |
| mempool | projected_first_block_fee_range_high | 2669 | 656 | 486 | varying |
| news-gdelt | crypto_terms_per_million | 183 | 178 | 179 | varying |
| news-gdelt | story_count | 183 | 154 | 18 | varying |
| news-gdelt | source_count | 183 | 151 | 12 | varying |
| news-gdelt | mean_tone | 183 | 178 | 177 | varying |
| news-gdelt | mean_positive | 183 | 178 | 177 | varying |
| news-gdelt | mean_negative | 183 | 178 | 176 | varying |
| news-gdelt | mean_polarity | 183 | 178 | 177 | varying |

## Predictive smoke/early screen

Scores are out-of-sample gain over trailing-return and trailing-volatility bins. Positive bits/target is better. The first 60% of this live window fits all quantile edges and categorical probabilities; the last 40% is evaluated in six chronological blocks.
This is a single chronological split. During the smoke stage, a few appended observations can move the split boundary and materially reorder sparse categorical scores; do not treat the ranking as a selected model basis until it survives separated-day evaluation.

| family | feature | target | eval rows | effective outcomes | bits/target | positive blocks | decision | evidence |
|---|---|---:|---:|---:|---:|---:|---|---|
| news-gdelt | mean_tone | 30m | 45 | 22 | 0.214526 | 4/5 | promising | early |
| options | put_call_25d_skew_7d | 30m | 665 | 22 | 0.184676 | 5/6 | promising | early |
| news-gdelt | mean_negative | 30m | 45 | 22 | 0.156320 | 2/5 | inconclusive | early |
| mempool | half_hour_fee | 15m | 786 | 52 | 0.067119 | 5/6 | promising | early |
| news-gdelt | mean_tone | 1m | 69 | 69 | 0.066352 | 4/6 | inconclusive | early |
| news-gdelt | crypto_terms_per_million | 30m | 45 | 22 | 0.066307 | 4/5 | promising | early |
| mempool | total_fee_btc | 15m | 786 | 52 | 0.063171 | 4/6 | inconclusive | early |
| news-gdelt | source_count | 5s | 71 | 71 | 0.062633 | 4/6 | inconclusive | early |
| futures-premium | funding_rate_bps | 1m | 1031 | 1031 | 0.050166 | 4/6 | inconclusive | early |
| mempool | mean_fee_sat_vbyte | 15m | 786 | 52 | 0.049178 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_7d | 15s | 1055 | 1055 | 0.048655 | 5/6 | promising | early |
| news-gdelt | mean_polarity | 1s | 74 | 74 | 0.047701 | 4/6 | inconclusive | early |
| futures-premium | funding_rate_bps | 15s | 1055 | 1055 | 0.046195 | 5/6 | promising | early |
| options | put_call_25d_skew_30d | 15s | 1055 | 1055 | 0.045420 | 6/6 | promising | early |
| options | put_call_25d_skew_7d | 15m | 786 | 52 | 0.041839 | 4/6 | inconclusive | early |
| options | atm_iv_term_7d_minus_1d | 1m | 1030 | 1030 | 0.039496 | 6/6 | promising | early |
| mempool | total_fee_btc | 30m | 665 | 22 | 0.035597 | 3/6 | inconclusive | early |
| mempool | virtual_size | 15m | 786 | 52 | 0.035228 | 5/6 | promising | early |
| options | put_call_25d_skew_1d | 15s | 1055 | 1055 | 0.033382 | 5/6 | promising | early |
| news-gdelt | mean_negative | 1s | 74 | 74 | 0.033279 | 4/6 | inconclusive | early |
| options | atm_iv_1d | 5s | 1063 | 1063 | 0.031841 | 4/6 | inconclusive | early |
| news-gdelt | mean_positive | 1s | 74 | 74 | 0.031234 | 5/6 | promising | early |
| options | put_call_25d_skew_1d | 1m | 1030 | 1030 | 0.028939 | 4/6 | inconclusive | early |
| news-gdelt | mean_tone | 1s | 74 | 74 | 0.028001 | 4/6 | inconclusive | early |
| options | put_call_25d_skew_1d | 5s | 1063 | 1063 | 0.026402 | 5/6 | promising | early |
| options | put_call_25d_skew_30d | 1m | 1030 | 1030 | 0.023291 | 5/6 | promising | early |
| news-gdelt | story_count | 5s | 71 | 71 | 0.020092 | 2/6 | inconclusive | early |
| options | put_call_25d_skew_1d | 15m | 786 | 52 | 0.019747 | 4/6 | inconclusive | early |
| options | atm_iv_1d | 15s | 1055 | 1055 | 0.018660 | 4/6 | inconclusive | early |
| options | atm_iv_1d | 15m | 786 | 52 | 0.018155 | 3/6 | inconclusive | early |

## Retention decision

1. Stop a feed immediately only if it is malformed, timestamp-misaligned, stale, or near-constant.
2. At 24h, promote repeated positive results to `early`; do not reject a varying feature from a single day.
3. Starting with separated days, stop a feature family only when its upper uncertainty bound is below the chosen material gain threshold across target horizons.
4. Compact high-volume raw books into causal 1s derived features after their reconstruction tests; the unresolved slow feeds themselves are cheap enough to retain through the required evidence window.

Machine-readable results are in `data/benchmarks/live-external-early-screen.json`.
