# Binance cross-product portfolio basis

The portfolio-basis experiment selects actual Binance assets whose return
vectors add the most independent information to a growing set. It is a market
structure tool, not a trading strategy or a portfolio weighting rule.

## Run

```bash
npm run basis:binance
```

The default run:

- discovers active listings across Spot, USD-M perpetuals (including TradFi),
  COIN-M perpetuals, and Options;
- maps contracts to economic underlyings and deduplicates the same asset across
  products;
- uses Spot as the canonical price when possible, then USD-M, then COIN-M;
- uses a 365-day window of daily candles by default, while `--interval` can
  select aligned intraday candles;
- excludes stable assets and leveraged tokens, requires trades or quote volume
  on at least half the UTC days, and rejects price series stale for more than
  48 active hours;
- anchors the selection with BTC and grows the basis until the median market
  projection R² reaches 80% and its 10th percentile reaches 50%, capped at 512
  assets;
- treats pivots within 5% of the maximum unexplained variance as
  near-equivalent and chooses the largest mean absolute candle return within
  that set; and
- writes JSON and Markdown reports under `docs/portfolio/`, while reusable
  market-data caches remain under `data/portfolio-basis/`.

Useful options:

```bash
npm run basis:binance -- --size 24
npm run basis:binance -- --target-median-r2 0.9 --target-p10-r2 0.7
npm run basis:binance -- --method spearman
npm run basis:binance -- --anchor none
npm run basis:binance -- --min-median-quote-volume 1000000
npm run basis:binance -- --products usdm-futures
npm run basis:binance -- --end 2026-06-30 --days 365
npm run basis:binance -- --days 30 --interval 4h
npm run basis:binance -- --candles 360 --interval 15m
npm run basis:binance -- --residual-equivalence-band 0.02
```

Run with `--help` for the complete option list.

`--days` controls the UTC-day lookback window; `--interval` controls sampling.
For example, `--days 30 --interval 4h` produces 180 aligned four-hour returns
from 181 closes. Intraday candles are cached separately by interval, and large
requests are paginated at Binance's 1,000-candle response limit.

`--candles` requests an exact number of aligned returns and is mutually
exclusive with `--days`. This is the preferred mode for comparing intervals:
360 samples correspond to 360 days at 1d, 60 days at 4h, 15 days at 1h, 3.75
days at 15m, and 6 hours at 1m.

Eligibility is interval-neutral: complete candles are required, but zero returns
are retained because unchanged closes are legitimate observations. Trading
activity is measured by trades or quote volume per UTC day, liquidity is
aggregated to daily volume before applying a threshold, and only long consecutive
stale-price runs are rejected. Zero-volume off-session TradFi candles are ignored
when measuring stale runs.

## Why pivoted QR

Let each column of a matrix be one asset's centered, normalized return series.
The dot product of two columns is their Pearson correlation. Column-pivoted QR
repeatedly selects the asset with the largest residual after projecting every
remaining asset onto the span of the already selected assets.

This is stronger than greedily finding small pairwise correlations. An asset can
have moderate correlation to each of two selected assets yet be almost entirely
explained by their linear combination. Its QR residual will correctly be small.
Positive and negative correlations are both treated as dependence because the
projection is squared; an asset with correlation `-1` is redundant, not
orthogonal.

## Return-amplitude tie-break

Pure pivoted QR can choose a quiet asset even when another asset contributes
almost the same independent direction with larger candles. The default
selector therefore uses a two-level objective:

1. find the maximum remaining unexplained variance;
2. among candidates within 5% of that maximum, select the largest mean absolute
   candle return.

The comparison is always made inside one interval. Daily and minute return
magnitudes are never mixed. Set `--residual-equivalence-band 0` to recover pure
maximum-residual QR.

The report records mean absolute return for every eligible asset and displays
it in basis points for selected assets. This metric measures movement, not
tradeable edge. A liquidity floor remains important because volatile,
thinly-traded assets can otherwise dominate the tie-break.

The report includes:

- residual ratio at each selection step;
- maximum and mean absolute pairwise correlation;
- a selected-asset correlation matrix;
- projection R² for every eligible market asset; and
- the least-covered assets, which indicate whether the requested basis size is
  too small.

It also distinguishes *market listings* from *economic assets*. The production
catalog contains thousands of rows because it includes every quote pair,
inactive Spot pair, dated/settling contract, and individual option
strike/expiry. Those rows must not become separate columns in an asset-return
basis. The report preserves the full listing counts while separately reporting
active, deduplicated underlyings and eligible aligned return series.

## Product mapping

Spot, USD-M, and COIN-M listings provide continuous candle histories. Options
contribute their underlying/product provenance but individual strikes and
expiries are not treated as separate assets. Their transient contract returns
are not comparable to a continuous one-year asset return.

Prediction outcomes are deliberately excluded. They are bounded binary claims
with token order books rather than continuous asset candle histories.

When `--products all` is used, Spot and USD-M returns are USDT-denominated while
COIN-M fallback series are USD-denominated. This makes otherwise unavailable
COIN-M underlyings usable, but introduces small USD/USDT basis risk.

## Interpretation and limitations

The default result is intentionally a *descriptive orthogonal basis*. It tends
to select unusual assets with idiosyncratic return histories. It should not be
interpreted as an equal-weight portfolio.

Before allocating capital:

1. impose a liquidity floor appropriate to execution;
2. repeat the analysis over adjacent and rolling windows;
3. measure selection frequency and out-of-sample correlation stability;
4. decide the basis size from whole-market projection coverage; and
5. apply a separate risk/weighting model such as inverse volatility, HRP, or a
   constrained maximum-diversification optimizer.

Correlation networks and community selection are established alternatives.
They are useful when the goal is one representative per market cluster. QR
column selection is used here because the stated goal is actual assets forming
a near-orthogonal spanning set.

## References

- [Official Binance Spot REST API](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md)
- [Official Binance public-data archive](https://github.com/binance/binance-public-data)
- [Tropp: Column Subset Selection, Matrix Factorization, and Eigenvalue Optimization](https://arxiv.org/abs/0806.4404)
- [López de Prado: Building Diversified Portfolios that Outperform Out of Sample](https://papers.ssrn.com/sol3/abstract_id=2708678)
- [Jing and Correa Rocha: A network-based strategy of price correlations for optimal cryptocurrency portfolios](https://arxiv.org/abs/2304.02362)
- [Gavin and Crane: Community Detection in Cryptocurrencies with Potential Applications to Portfolio Diversification](https://arxiv.org/abs/2108.09763)
