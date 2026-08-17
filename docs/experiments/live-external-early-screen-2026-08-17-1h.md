# Live external-feature early screen — 2026-08-17

Generated at 2026-08-17T10:44:02.185Z after **1.000 hours** of collection.

**Current interpretation:** Pipeline and very-large-effect smoke test only. Negative results cannot reject a feature.

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

- Collected so far: 37.65 MiB; projected total: 903.61 MiB/day.
- Unresolved slow candidate feeds: 0.61 MiB so far; projected 14.65 MiB/day.
- Candidate bytes are the unresolved options, premium, GDELT, and mempool feeds. High-volume exchange books are excluded because sparse Tardis evidence has already shown short-horizon value.

## Feed inventory

| source | files | stored | projected/day |
|---|---:|---:|---:|
| coinbase-btcusd-level2 | 2 | 10.05 MiB | 241.20 MiB |
| kraken-btcusd-book | 2 | 8.95 MiB | 214.73 MiB |
| binance-usdm-book-ticker | 2 | 7.86 MiB | 188.59 MiB |
| binance-spot-depth-diff | 2 | 5.45 MiB | 130.90 MiB |
| deribit-btc-perpetual-book | 2 | 2.91 MiB | 69.90 MiB |
| binance-spot-aggtrade | 2 | 1.16 MiB | 27.78 MiB |
| binance-spot-depth-snapshot | 2 | 0.66 MiB | 15.75 MiB |
| deribit-btc-option-surface-raw | 8 | 0.24 MiB | 5.75 MiB |
| mempool-live | 2 | 0.22 MiB | 5.32 MiB |
| deribit-btc-option-summary | 2 | 0.13 MiB | 3.02 MiB |
| binance-usdm-premium-index | 2 | 0.01 MiB | 0.31 MiB |
| gdelt-crypto-news | 2 | 0.01 MiB | 0.25 MiB |
| collector-session | 2 | 0.00 MiB | 0.02 MiB |
| binance-usdm-book-ticker-session | 2 | 0.00 MiB | 0.01 MiB |
| binance-spot-depth-diff-session | 2 | 0.00 MiB | 0.01 MiB |
| binance-spot-aggtrade-session | 2 | 0.00 MiB | 0.01 MiB |
| binance-usdm-mark-price-session | 2 | 0.00 MiB | 0.01 MiB |
| binance-usdm-liquidations-session | 2 | 0.00 MiB | 0.01 MiB |
| coinbase-btcusd-level2-session | 2 | 0.00 MiB | 0.01 MiB |
| deribit-btc-perpetual-book-session | 2 | 0.00 MiB | 0.01 MiB |
| kraken-btcusd-book-session | 2 | 0.00 MiB | 0.01 MiB |

## Candidate feature health

| family | feature | observations | changes | distinct | health |
|---|---|---:|---:|---:|---|
| futures-premium | premium_bps | 60 | 59 | 60 | varying |
| futures-premium | funding_rate_bps | 60 | 40 | 41 | varying |
| futures-premium | mark_index_abs_gap_bps | 60 | 59 | 60 | varying |
| options | atm_iv_1d | 60 | 52 | 50 | varying |
| options | atm_iv_7d | 60 | 51 | 39 | varying |
| options | atm_iv_30d | 60 | 25 | 12 | varying |
| options | atm_iv_term_7d_minus_1d | 60 | 50 | 39 | varying |
| options | atm_iv_term_30d_minus_7d | 60 | 51 | 41 | varying |
| options | put_call_25d_skew_1d | 60 | 59 | 55 | varying |
| options | put_call_25d_skew_7d | 60 | 43 | 22 | varying |
| options | put_call_25d_skew_30d | 60 | 40 | 25 | varying |
| options | total_call_put_oi_imbalance | 60 | 45 | 46 | varying |
| options | one_day_call_put_oi_imbalance | 60 | 18 | 19 | varying |
| options | log_distance_to_major_strike | 60 | 59 | 58 | varying |
| options | hours_to_next_expiry | 60 | 59 | 60 | varying |
| mempool | transaction_count | 60 | 59 | 59 | varying |
| mempool | virtual_size | 60 | 59 | 60 | varying |
| mempool | total_fee_btc | 60 | 59 | 60 | varying |
| mempool | mean_fee_sat_vbyte | 60 | 59 | 60 | varying |
| mempool | fastest_fee | 60 | 7 | 3 | varying |
| mempool | half_hour_fee | 60 | 0 | 1 | near-constant |
| mempool | hour_fee | 60 | 0 | 1 | near-constant |
| mempool | economy_fee | 60 | 0 | 1 | near-constant |
| mempool | minimum_fee | 60 | 0 | 1 | near-constant |
| mempool | projected_first_block_vsize | 60 | 58 | 54 | varying |
| mempool | projected_first_block_fee_range_high | 60 | 15 | 15 | varying |
| news-gdelt | crypto_terms_per_million | 4 | 3 | 4 | varying |
| news-gdelt | story_count | 4 | 3 | 4 | varying |
| news-gdelt | source_count | 4 | 3 | 4 | varying |
| news-gdelt | mean_tone | 4 | 3 | 4 | varying |
| news-gdelt | mean_positive | 4 | 3 | 4 | varying |
| news-gdelt | mean_negative | 4 | 3 | 4 | varying |
| news-gdelt | mean_polarity | 4 | 3 | 4 | varying |

## Predictive smoke/early screen

Scores are out-of-sample gain over trailing-return and trailing-volatility bins. Positive bits/target is better. The first 60% of this live window fits all quantile edges and categorical probabilities; the last 40% is evaluated in six chronological blocks.

| family | feature | target | eval rows | effective outcomes | bits/target | positive blocks | decision | evidence |
|---|---|---:|---:|---:|---:|---:|---|---|
| futures-premium | premium_bps | 5m | 20 | 3 | 0.593106 | 2/2 | promising | smoke-only |
| futures-premium | mark_index_abs_gap_bps | 5m | 20 | 3 | 0.593106 | 2/2 | promising | smoke-only |
| options | total_call_put_oi_imbalance | 5m | 20 | 3 | 0.558319 | 2/2 | promising | smoke-only |
| options | one_day_call_put_oi_imbalance | 5m | 20 | 3 | 0.544563 | 2/2 | promising | smoke-only |
| mempool | projected_first_block_vsize | 5m | 20 | 3 | 0.525860 | 2/2 | promising | smoke-only |
| options | atm_iv_term_7d_minus_1d | 5m | 20 | 3 | 0.432226 | 2/2 | promising | smoke-only |
| futures-premium | funding_rate_bps | 5m | 20 | 3 | 0.382226 | 2/2 | promising | smoke-only |
| options | hours_to_next_expiry | 5m | 20 | 3 | 0.382226 | 2/2 | promising | smoke-only |
| mempool | fastest_fee | 1m | 24 | 23 | 0.365336 | 3/3 | promising | smoke-only |
| options | put_call_25d_skew_7d | 15s | 24 | 24 | 0.335338 | 3/3 | promising | smoke-only |
| mempool | projected_first_block_vsize | 15s | 24 | 24 | 0.316451 | 3/3 | promising | smoke-only |
| options | total_call_put_oi_imbalance | 15s | 24 | 24 | 0.310285 | 3/3 | promising | smoke-only |
| mempool | virtual_size | 1m | 24 | 23 | 0.307123 | 3/3 | promising | smoke-only |
| options | put_call_25d_skew_30d | 1m | 23 | 22 | 0.299489 | 2/2 | promising | smoke-only |
| options | atm_iv_term_30d_minus_7d | 1m | 23 | 22 | 0.289976 | 2/2 | promising | smoke-only |
| options | log_distance_to_major_strike | 5m | 20 | 3 | 0.279551 | 2/2 | promising | smoke-only |
| mempool | fastest_fee | 5m | 20 | 3 | 0.270866 | 1/2 | inconclusive | smoke-only |
| options | put_call_25d_skew_30d | 5m | 20 | 3 | 0.266467 | 2/2 | promising | smoke-only |
| options | log_distance_to_major_strike | 15s | 24 | 24 | 0.252554 | 2/3 | inconclusive | smoke-only |
| mempool | projected_first_block_fee_range_high | 1m | 24 | 23 | 0.246388 | 3/3 | promising | smoke-only |
| options | put_call_25d_skew_30d | 15s | 24 | 24 | 0.240168 | 1/3 | inconclusive | smoke-only |
| mempool | total_fee_btc | 1m | 24 | 23 | 0.229492 | 3/3 | promising | smoke-only |
| mempool | mean_fee_sat_vbyte | 1m | 24 | 23 | 0.229492 | 3/3 | promising | smoke-only |
| options | put_call_25d_skew_30d | 1s | 24 | 24 | 0.223538 | 3/3 | promising | smoke-only |
| options | atm_iv_1d | 15s | 24 | 24 | 0.213003 | 2/3 | inconclusive | smoke-only |
| mempool | transaction_count | 1m | 24 | 23 | 0.202665 | 3/3 | promising | smoke-only |
| mempool | projected_first_block_fee_range_high | 5m | 20 | 3 | 0.198404 | 1/2 | inconclusive | smoke-only |
| futures-premium | funding_rate_bps | 1m | 23 | 22 | 0.191027 | 2/2 | promising | smoke-only |
| options | hours_to_next_expiry | 1m | 23 | 22 | 0.191027 | 2/2 | promising | smoke-only |
| options | atm_iv_term_7d_minus_1d | 15s | 24 | 24 | 0.179152 | 2/3 | inconclusive | smoke-only |

## Retention decision

1. Stop a feed immediately only if it is malformed, timestamp-misaligned, stale, or near-constant.
2. At 24h, promote repeated positive results to `early`; do not reject a varying feature from a single day.
3. Starting with separated days, stop a feature family only when its upper uncertainty bound is below the chosen material gain threshold across target horizons.
4. Compact high-volume raw books into causal 1s derived features after their reconstruction tests; the unresolved slow feeds themselves are cheap enough to retain through the required evidence window.

Machine-readable results are in `data/benchmarks/live-external-early-screen.json`.
