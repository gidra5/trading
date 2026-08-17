# Volume-imbalance and volatility-index audit

Date: 2026-08-17  
Objective: determine whether signed volume imbalance, order-book imbalance, BTC DVOL, or the Cboe VIX should enter the causal BTCUSDT return-distribution basis.

## Definitions

Trade-volume imbalance uses executed aggressor flow:

$$
I^{trade}_t=\frac{V^{taker\ buy}_t-V^{taker\ sell}_t}{V^{taker\ buy}_t+V^{taker\ sell}_t}.
$$

Use quote-notional volume for BTCUSDT so trades at different prices are comparable. The corresponding trade-count imbalance replaces volumes by buyer- and seller-initiated trade counts.

Order-book volume imbalance uses resting liquidity:

$$
I^{book}_{t,L}=\frac{\sum_{i=1}^{L}Q^{bid}_{t,i}-\sum_{i=1}^{L}Q^{ask}_{t,i}}{\sum_{i=1}^{L}Q^{bid}_{t,i}+\sum_{i=1}^{L}Q^{ask}_{t,i}}.
$$

These are not interchangeable. Trade imbalance measures executed pressure; book imbalance measures the current queue shape and can disappear through cancellations rather than trades.

VIX is the Cboe daily 30-day S&P 500 option-implied volatility index. DVOL is Deribit's BTC option-implied volatility index. Both are volatility-regime variables, not directional signals by construction.

## Imbalance results

The 90-day forward-market screen conditions on the existing price/volume/range basis. The later 30-day global screen searches the broader feature universe jointly.

| target | coordinate | lookback/delay | primary bits | transfer bits | decision |
|---:|---|---|---:|---:|---|
| 1s | spot aggregate-trade count imbalance | latest completed 1s | 0.039090 | 0.047475 | stable in the 90-day screen |
| 1s | spot taker quote-volume imbalance | latest completed 1s | 0.023082 | 0.036053 | stable in the 90-day screen |
| 1s | spot raw-trade imbalance EMA | 2s | 0.010400 | 0.016225 | stable compact smoother |
| 1s | aggregate-count imbalance in the 30-day global search | latest completed 1s | 0.113532 / 0.013598 conditional | 0.105199 / 0.044549 conditional | selected jointly |
| 1s | spot L1 book quantity imbalance, dedicated fresh-book audit | latest snapshot strictly before origin, maximum age 5s | 0.014954 | 0.022972 | positive in all four dedicated sub-blocks |
| 1m | spot raw-trade imbalance EMA | 8s | 0.035626 standalone | 0.039633 standalone | positive alone, not selected jointly |
| 15m | spot taker quote imbalance EMA | 128s | 0.010187 standalone | 0.017849 standalone | unstable: one primary half negative |
| 1h | best tested imbalance coordinate | latest/EMA/book variants | negative | negative | reject |

The dedicated spot-book test found a monotone active-sign relation: from the most ask-heavy to the most bid-heavy L1 quartile, the conditional positive probability increased from 37.222% to 62.713%. However, the later common-coverage 30-day joint screen gave L1 book imbalance 0.017525 primary but -0.002830 transfer bits. Keep book imbalance provisional and gated until a longer continuous book history resolves this split difference.

### Input decision

- Required for the 1s flow branch: last aggressor side, quote-volume imbalance, and trade-count imbalance for the latest completed second. A model with raw buy/sell totals can derive the ratios itself.
- Optional 1s smoother: 2s and 8s EMAs of trade imbalance. Do not supply a large bank of overlapping EMAs by default.
- Provisional 1s book branch: L1 and top-5 quantity imbalance, microprice offset, source age, and observed mask. Gate or drop the branch when the snapshot is older than 5s.
- For the 1m head, retain the 8s trade-imbalance EMA only as an ablation input; it did not survive the joint subset selection.
- Omit imbalance from the required 15m and 1h heads. Its directional lifetime is measured in seconds, not hours.

## Volatility-index results

The DVOL backfill contains 47,338 hourly observations. The VIX backfill contains 1,387 valid daily closes from 2021-03-24 through 2026-08-17, sourced from FRED's Cboe VIXCLS series. Every daily VIX close is delayed until the following UTC day before it can enter a target.

All values below are held-out additions after the same-horizon BTC trailing return and realized-volatility state. Negative values mean the extra conditional cells forecast worse out of sample; they are evidence against promotion, not negative population mutual information.

| target | best VIX candidate | first primary half | second primary half | transfer | selected |
|---:|---|---:|---:|---:|:---:|
| 1m | VIX minus 7d BTC realized volatility | -0.006488 | -0.035174 | -0.001802 | no |
| 5m | 21-session VIX change | -0.040192 | -0.029401 | -0.040797 | no |
| 15m | 5-session VIX change | -0.021542 | -0.045237 | -0.019594 | no |
| 30m | VIX minus 7d BTC realized volatility | -0.015868 | -0.034663 | -0.027522 | no |
| 1h | 21-session VIX change | -0.033780 | -0.044182 | -0.040747 | no |

| target | best DVOL candidate | first primary half | second primary half | transfer | selected |
|---:|---|---:|---:|---:|:---:|
| 5m | BTC DVOL level | -0.030162 | -0.050221 | -0.044626 | no |
| 15m | BTC DVOL level | -0.057820 | -0.045841 | -0.015872 | no |
| 30m | BTC DVOL level | -0.023896 | -0.026771 | -0.027649 | no |
| 1h | BTC DVOL level | -0.053697 | -0.026958 | -0.020236 | no |

### Interpretation and decision

VIX is a slow US-equity risk-regime measurement. At a 1h-or-shorter BTC horizon it is stale, market-specific, and mostly dominated by BTC's own current realized volatility and cross-crypto volatility state. DVOL is semantically closer to BTC, but its level and changes still add no stable information after that BTC state.

Do not add daily VIX or hourly DVOL to the current required model basis. They may still be useful for multi-day risk limits, leverage scaling, or stress-state labels, which is a different target from forecasting the next 1m–1h return distribution.

The rejection does not cover the full live Deribit option surface. ATM IV by tenor, 25-delta skew, skew changes, term curvature, strike/open-interest concentration, and expiry proximity can contain information that a single DVOL number discards. Continue collecting those features until there are enough independent chronological blocks for a separate test.

## Reproducibility

```text
npm run fetch:external-public -- --source cboe-vix --start 2021-03-24 --end 2026-08-17
npm run analysis:external-public
npm run test:external-public
npm run analysis:external-public:test
```

Machine-readable results are stored in `data/benchmarks/public-external-feature-information.json`. Supporting imbalance results are in `data/benchmarks/forward-market-return-information.json`, `data/benchmarks/spot-order-book-return-information.json`, and `data/benchmarks/global-return-feature-basis-30d.json`.
