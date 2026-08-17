# Public external feature information audit

Generated 2026-08-17T18:40:48.494Z. All scores are held-out increments beyond BTC's own trailing return and realized-volatility state.

## Outcome

The public backfill produced 47,338 hourly DVOL rows, 1,387 daily VIX rows, 1,972 Coin Metrics rows, 1,360 community whale/miner/derivatives rows, and 2,195 mined-block proxy rows. The matched cross-market corpus contains BTC plus ETHUSDT, SOLUSDT, BNBUSDT, DOGEUSDT minute bars.

5 conditional coordinates passed the three-block stability rule across all source/horizon screens. A selected coordinate is a distribution feature, not automatically a profitable direction signal.

## Selected conditional basis

| screen | horizon | step | feature | lookback | primary bits | transfer bits | sign bits | magnitude bits |
|---|---:|---:|---|---|---:|---:|---:|---:|
| cross-market-1m | 1m | 1 | ETH realized volatility | 30m | 0.069524 | 0.036094 | 0.000643 | 0.068409 |
| cross-market-1m | 1m | 2 | ETH/BTC volatility ratio | 30m | 0.024795 | 0.034725 | 0.000453 | 0.025148 |
| cross-market-1m | 1m | 3 | ETH realized volatility | 60m | 0.000809 | 0.002094 | -0.000525 | 0.002918 |
| cross-market-5m | 5m | 1 | ETH realized volatility | 60m | 0.026971 | 0.017613 | -0.000472 | 0.029268 |
| cross-market-15m | 15m | 1 | ETH realized volatility | 60m | 0.010539 | 0.005732 | -0.000329 | 0.012938 |

## Best marginal candidate by screen

| screen | candidates | best candidate | stable | lookback | first half | second half | transfer |
|---|---:|---|:---:|---|---:|---:|---:|
| cross-market-1m | 104 | ETH realized volatility | yes | 30m | 0.075462 | 0.063778 | 0.036094 |
| cross-market-5m | 104 | ETH realized volatility | yes | 60m | 0.037844 | 0.016449 | 0.017613 |
| cross-market-15m | 104 | ETH realized volatility | yes | 60m | 0.014857 | 0.006360 | 0.005732 |
| cross-market-30m | 104 | SOL realized volatility | no | 5m | -0.000645 | 0.000014 | 0.000087 |
| cross-market-60m | 104 | ETH/BTC volatility ratio | no | 5m | -0.001630 | -0.002801 | -0.002705 |
| deribit-dvol-5m | 17 | BTC DVOL level | no | latest completed 1h | -0.030162 | -0.050221 | -0.044626 |
| deribit-dvol-15m | 17 | BTC DVOL level | no | latest completed 1h | -0.057820 | -0.045841 | -0.015872 |
| deribit-dvol-30m | 17 | BTC DVOL level | no | latest completed 1h | -0.023896 | -0.026771 | -0.027649 |
| deribit-dvol-60m | 17 | BTC DVOL level | no | latest completed 1h | -0.053697 | -0.026958 | -0.020236 |
| cboe-vix-1m | 9 | VIX minus BTC realized volatility | no | 7d BTC realized | -0.006488 | -0.035174 | -0.001802 |
| cboe-vix-5m | 9 | VIX change | no | 21 trading day(s) | -0.040192 | -0.029401 | -0.040797 |
| cboe-vix-15m | 9 | VIX change | no | 5 trading day(s) | -0.021542 | -0.045237 | -0.019594 |
| cboe-vix-30m | 9 | VIX minus BTC realized volatility | no | 7d BTC realized | -0.015868 | -0.034663 | -0.027522 |
| cboe-vix-60m | 9 | VIX change | no | 21 trading day(s) | -0.033780 | -0.044182 | -0.040747 |
| coinmetrics-15m | 37 | Total-fee change | no | 1d | -0.031710 | -0.024359 | -0.032310 |
| coinmetrics-30m | 37 | Hash-rate change | no | 7d | -0.042535 | -0.043981 | -0.037322 |
| coinmetrics-60m | 37 | Exchange inflow | no | 3d sum | -0.040880 | -0.035758 | -0.039153 |
| community-daily-15m | 145 | exchange stablecoins ratio usd change | no | 3d | -0.025313 | -0.020349 | -0.024508 |
| community-daily-30m | 145 | funding rates change | no | 30d | -0.025050 | -0.038036 | -0.026301 |
| community-daily-60m | 145 | exchange stablecoins ratio usd change | no | 1d | -0.036763 | -0.036480 | -0.037021 |
| mempool-proxy-15m | 36 | Median block fee rate change | no | 14 buckets | -0.030885 | -0.014173 | -0.029612 |
| mempool-proxy-30m | 36 | 90th-percentile block fee rate change | no | 2 buckets | -0.032752 | -0.029570 | -0.029889 |
| mempool-proxy-60m | 36 | Average block weight change | no | 6 buckets | -0.039481 | -0.037081 | -0.037786 |
| joint-external-60m | 348 | Alt-market return dispersion | no | 30m | -0.021137 | -0.031820 | -0.021670 |

## Interpretation constraints

- DVOL is an hourly volatility-index history, not a historical full option surface; ATM, 25-delta skew, term structure, strike OI, and IV/skew changes remain live-forward measurements.
- VIX is a daily US-equity option-implied volatility index. The backtest delays each close until the next UTC day and therefore tests it as a slow macro regime feature, not a live intraday VIX feed.
- mempool.space history contains mined-block aggregates rather than the earlier unconfirmed transaction backlog; it is labeled as a mempool proxy and not as historical live mempool state.
- Coin Metrics Community exchange-flow history is downloaded retrospectively and can be revised. The screen assumes next-day availability and stores each row's latest revision time separately; results are provisional and not a true point-in-time test.
- The community whale/miner/liquidation/derivatives archive is CC-BY but upstream-derived and retrospectively revised. Its next-day screen is exploratory, not point-in-time production evidence.
- The cross-market screen covers liquid crypto spot markets. CME macro futures still require a licensed intraday point-in-time source.
- Information gain measures distribution forecast value before fees, latency, market impact, or a trading decision rule.

## Method

- Target: Eight training-quantile cells of the forward BTC log return.
- Baseline: Quartiles of the same-horizon trailing BTC return crossed with trailing realized volatility.
- Candidate: Training-only quartiles; missing observations are omitted on a matched basis.
- Score: Held-out candidate-minus-baseline log likelihood in bits per target, plus sign and magnitude components.
- Stability: Positive full-distribution gain in both chronological primary halves and the separated transfer block.
- Selection: At each step choose the candidate with the largest positive worst-block gain conditional on the already selected quartile coordinates.
- Slow sources: Hourly evaluation to avoid treating repeated daily or half-daily values as independent minute observations.

Complete candidate rankings, frozen quantile edges, observation counts, sign decomposition, and magnitude decomposition are in `data/benchmarks/public-external-feature-information.json`.
