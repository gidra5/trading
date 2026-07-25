# Binance independent-scale basis comparison

The multiscale experiment builds and compares five independent Binance market
bases. Every scale uses exactly 360 return observations:

| Scale | Approximate horizon | Role |
| --- | ---: | --- |
| 1d | 360 days | structural regime |
| 4h | 60 days | swing structure |
| 1h | 15 days | adaptive trading structure |
| 15m | 3.75 days | intraday structure |
| 1m | 6 hours | execution and microstructure |

Equal sample counts keep the matrix rank and estimator sample size comparable.
They do not make the economic horizons equivalent: finer scales still contain
more asynchronous trading and microstructure noise.

## Run

Generate the five source reports with a common end day:

```bash
npm run basis:binance -- --candles 360 --interval 1d
npm run basis:binance -- --candles 360 --interval 4h
npm run basis:binance -- --candles 360 --interval 1h
npm run basis:binance -- --candles 360 --interval 15m
npm run basis:binance -- --candles 360 --interval 1m
```

Then compare them:

```bash
npm run basis:compare-scales
```

The comparison loads the latest complete q4 report for each scale, reconstructs
a pure maximum-residual QR basis at the same size, and reports:

- return-prioritized versus pure-QR basis overlap;
- mean and median absolute-return uplift;
- changes in projection coverage and selection residual;
- pairwise basis overlap between intervals;
- assets selected at every scale and at a majority of scales; and
- per-scale coverage of the simple 3-of-5 membership consensus.

Reports are written under `data/portfolio-basis/scale-comparison/`.

## Point-in-time index backtest

The static comparison is descriptive. It must not be backcast by applying its
final constituents or weights to earlier prices. The investable index instead
re-evaluates every scale using only information available at each timestamp:

```bash
npm run basis:backtest-index
```

At minute close `t`:

1. use only candles completed by `t`;
2. rebuild the 1m sleeve, and rebuild 15m, 1h, 4h, or 1d sleeves whenever a
   new candle at that native scale has completed;
3. select each scale independently from its latest 360 native returns;
4. cap and weight that scale from its own trailing quote notional;
5. combine five 20% sleeves; and
6. trade the new target at `t`, then apply it only to the `t → t+1` candle.

Between native closes, re-evaluating a slower sleeve would produce the same
inputs and target, so its last point-in-time result is retained. The 1m sleeve
is genuinely rebuilt on every minute.

The historical universe comes from Binance Vision archive directories rather
than the current exchange catalog. An economic asset becomes eligible only
after a complete 360-return window exists at that timestamp. Spot is the first
canonical route, followed by USD-M and COIN-M when the earlier route is not
point-in-time eligible. Expiring options are catalogued and mapped to their
underlyings, but individual strikes and expiries are not durable continuous
return series. The raw product-row list is persisted separately so listing
counts are not confused with the smaller deduplicated set of economic return
series used by the basis.

Minute candles and a methodology/performance report are written under
`data/portfolio-basis/index-history/`.

### Long-only friction

The simulated portfolio has exposure in `[0, 1]`. It never shorts, borrows, or
uses leverage, and it charges no interest, maintenance, or borrow cost.
Missing or not-yet-warm sleeve weight remains cash rather than being
renormalized with hindsight. USD-M and COIN-M positions do pay or receive the
actual archived perpetual-funding rate at each settlement timestamp.

Every minute rebalance is self-financing. Post-cost risky weights equal the
new targets, while transaction cost is charged on the absolute notional of
every buy and every sell. The default stored net index uses:

- Spot: 10 bp fee plus 5 bp execution loss;
- USD-M/COIN-M: 5 bp fee plus 5 bp execution loss.

The output also retains gross, fee-only, and conservative 20 bp execution-loss
paths. These fixed-bp paths make assumptions auditable, but they are not a
capacity model. AUM and historical order-book depth are required to estimate
market impact.

## S&P-like ranking and weighting

Basis membership and index weighting are intentionally separate. Orthogonality,
movement, and coverage determine which assets belong to each scale basis. The
index layer then ranks and weights those already-selected assets.

The S&P analogue would be float-adjusted market capitalization. Binance's
official market-data endpoints do not provide a consistent circulating or
free-float supply field for the full Spot, USD-M, COIN-M, Options-underlying,
and TradFi universe. The static generated index therefore labels its size
measure explicitly as a proxy:

```text
investable size = median daily quote volume of the canonical market
```

The walk-forward index uses trailing quote notional across the same 360 native
candles instead. This is exactly point-in-time at every scale and avoids using
a completed future UTC day in an intraday rebalance. The common factor that
converts the rolling total to an average daily value does not change
proportional or capped weights.

For each scale:

1. rank selected assets by the size proxy;
2. calculate proportional weights;
3. cap every constituent at 5%;
4. redistribute excess weight proportionally among uncapped assets until the
   sleeve sums to 100%.

The final static multiscale weight is an equal-weight index of those five
independently constructed sleeves:

```text
final weight(asset) = 20% × Σ per-scale capped weight(asset, scale)
```

An asset absent from a scale contributes zero in that sleeve. In the current
static report the aggregate is a 437-asset union, while recurrent, liquid
assets naturally receive more weight. Walk-forward membership is allowed to
change at every information update. The static JSON report retains uncapped
weights, capped weights, liquidity ranks, per-scale weights, aggregate ranks,
top-10 concentration, and effective constituent counts. A companion
`*.weights.csv` contains the final aggregate rank and per-scale sleeve weights
for direct portfolio tooling.

Quote volume measures tradability and attention, not economic capitalization.
Futures turnover can also be inflated by leverage or short-lived speculation.
If reliable point-in-time circulating-supply data becomes available, replace
the proxy with `price × investable circulating supply`; the capping and
multiscale aggregation steps can remain unchanged.

## Independent return-priority rule

At every scale, pivoted QR first finds the maximum unexplained variance among
the unselected assets. Candidates within 5% of that maximum are treated as
near-equivalent for the current pivot. The candidate with the largest mean
absolute candle return at that same interval is selected. That movement score
is measured on native log returns before the vectors are centered and
normalized for the correlation calculation.

This is a lexicographic objective:

1. retain near-maximal orthogonality;
2. among near-equivalent choices, prefer more movement.

The 5% band was retained after sensitivity checks on the current five reports.
Relative to pure QR at the same basis size, it increased both mean and median
absolute return at every interval. The largest coverage change was less than
0.6 percentage points, and the largest mean selection-residual change was less
than 0.3 percentage points. The band remains configurable through
`--residual-equivalence-band`.

Return magnitudes are never normalized or averaged across intervals. A daily
absolute return and a one-minute absolute return answer different questions;
cross-scale comparison therefore uses membership recurrence only.

## Interpreting recurrence

The intersection of all five bases is a stable multiscale core, but it is not a
spanning basis. Likewise, the set selected by at least three scales is a simple
consensus diagnostic rather than a jointly optimized portfolio.

This distinction matters. On the current sample, the 3-of-5 set does not reach
the 80% median projection-R² target at every interval. Use the independent
per-scale bases when scale-specific market coverage is required.

Mean absolute return is a movement proxy, not expected profit. It can favor
jumpy or difficult-to-execute assets. Live allocation still requires explicit
liquidity, spread, slippage, leverage, capacity, and out-of-sample stability
constraints.

## References

- [S&P U.S. Indices Methodology](https://www.spglobal.com/spdji/en/methodology/article/sp-us-indices-methodology/)
- [S&P Index Mathematics Methodology](https://www.spglobal.com/spdji/en/methodology/article/index-mathematics-methodology/)
- [Binance Spot market-data documentation](https://developers.binance.com/en/docs/products/spot/rest-api)
