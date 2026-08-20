# Public external feature information audit

Generated 2026-08-19T19:12:27.564Z. All scores are held-out increments beyond BTC's own trailing return and realized-volatility state.

## Outcome

The public backfill produced 47,396 hourly DVOL rows, 1,389 daily VIX rows, 21,733 macro observations across 36 series, 5,925 BTCUSDT funding settlements, 1,974 Coin Metrics rows, 1,362 community whale/miner/derivatives rows, and 2,195 mined-block proxy rows. The matched cross-market corpus contains BTC plus ETHUSDT, SOLUSDT, BNBUSDT, DOGEUSDT minute bars.

14 conditional coordinates passed the three-block stability rule across all source/horizon screens. A selected coordinate is a distribution feature, not automatically a profitable direction signal.

6 macro coordinates pass in the recent 180-day-fit screen, but 0 macro/funding coordinates pass when trained on 2021-2024 and tested on both halves of 2025 plus the 2026 transfer block. Therefore the recent US, UK, euro-area, and China results are regime-dependent research candidates, not required production inputs.

No settled-funding transformation passes either window. Slow CPI, labor, production, and GDP levels have too few independent releases for a reliable 1m-1h decision even when the carried-forward state creates many target rows.

## Best recent non-US candidate by economy

This is a discovery view, not a production-selection table. `stable` here means positive in three recent chronological blocks; none of these economies has a candidate that also passes the separate long-history screen.

| economy | horizon | best candidate | stable | lookback | first half | second half | transfer |
|---|---:|---|:---:|---|---:|---:|---:|
| Euro area | 30m | Euro-area AAA 2-year yield absolute change | yes | 1 trading observation(s) | 0.186267 | 0.107515 | 0.113880 |
| United Kingdom | 15m | United Kingdom industrial production index change | yes | 3 monthly observation(s) | 0.098798 | 0.168376 | 0.038805 |
| China | 60m | China CPI year-over-year absolute change | yes | 1 monthly observation(s) | 0.011948 | 0.120407 | 0.061028 |
| Japan | 30m | Japan industrial production index change | no | 1 monthly observation(s) | -0.011357 | 0.055759 | 0.086778 |
| India | 1m | India short-term interest rate change | no | 3 monthly observation(s) | 0.015919 | -0.008916 | 0.060182 |
| Russia | 30m | Russia inflation year-over-year release age | no | latest conservatively available release | -0.013686 | 0.194505 | -0.021870 |

## Do the recent macro winners survive the established basis?

No. The table compares the same coordinate on the same 180-day/2026 windows. The second score appends it after training-median states of the established multiscale BTC volatility basis and the selected ETH-volatility coordinate through 15m.

| horizon | coordinate | simple primary / transfer | after basis primary / transfer | after-basis halves | survives |
|---:|---|---:|---:|---:|:---:|
| 5m | US 2-year Treasury yield absolute change, 21 trading observation(s) | 0.110301 / 0.111357 | -0.120609 / -0.068450 | -0.248977, 0.003618 | no |
| 15m | United Kingdom industrial production index change, 3 monthly observation(s) | 0.134157 / 0.038805 | -0.162372 / 0.053900 | -0.057080, -0.264267 | no |
| 30m | US 10-year minus 2-year yield spread change, 1 trading observation(s) | 0.150392 / 0.132182 | -0.109028 / -0.166329 | -0.065459, -0.151191 | no |
| 30m | Euro-area AAA 2-year yield absolute change, 1 trading observation(s) | 0.146246 / 0.113880 | -0.206629 / -0.251231 | -0.180560, -0.231857 | no |
| 60m | Euro-area AAA 2-year yield change, 1 trading observation(s) | 0.193317 / 0.015992 | 0.033865 / 0.049231 | 0.087029, -0.017583 | no |
| 60m | China CPI year-over-year absolute change, 1 monthly observation(s) | 0.067066 / 0.061028 | -0.146086 / -0.044399 | -0.214387, -0.079987 | no |

## Selected conditional basis

| screen | horizon | step | feature | lookback | primary bits | transfer bits | sign bits | magnitude bits |
|---|---:|---:|---|---|---:|---:|---:|---:|
| cross-market-1m | 1m | 1 | ETH realized volatility | 30m | 0.069524 | 0.036094 | 0.000643 | 0.068409 |
| cross-market-1m | 1m | 2 | ETH/BTC volatility ratio | 30m | 0.024795 | 0.034725 | 0.000453 | 0.025148 |
| cross-market-1m | 1m | 3 | ETH realized volatility | 60m | 0.000809 | 0.002094 | -0.000525 | 0.002918 |
| cross-market-5m | 5m | 1 | ETH realized volatility | 60m | 0.026971 | 0.017613 | -0.000472 | 0.029268 |
| cross-market-15m | 15m | 1 | ETH realized volatility | 60m | 0.010539 | 0.005732 | -0.000329 | 0.012938 |
| global-macro-5m | 5m | 1 | US 2-year Treasury yield absolute change | 21 trading observation(s) | 0.110301 | 0.111357 | -0.013318 | 0.019846 |
| global-macro-5m | 5m | 2 | US 2-year Treasury yield absolute change | 1 trading observation(s) | 0.113517 | 0.131069 | -0.113735 | 0.167396 |
| global-macro-15m | 15m | 1 | United Kingdom industrial production index change | 3 monthly observation(s) | 0.134157 | 0.038805 | -0.018174 | 0.067371 |
| global-macro-15m | 15m | 2 | US high-yield credit spread absolute change | 21 trading observation(s) | 0.027612 | 0.054415 | 0.026702 | -0.022424 |
| global-macro-30m | 30m | 1 | US 10-year minus 2-year yield spread change | 1 trading observation(s) | 0.150392 | 0.132182 | -0.064722 | -0.167696 |
| global-macro-60m | 60m | 1 | Euro-area AAA 2-year yield change | 1 trading observation(s) | 0.193317 | 0.015992 | -0.041463 | 0.224394 |
| global-macro-production-1m | 1m | 1 | US 10-year Treasury yield change | 21 trading observation(s) | 0.099376 | 0.147175 | -0.044819 | 0.089046 |
| global-macro-production-1m | 1m | 2 | Bank of Russia key rate change | 1 policy decision(s) | 0.031764 | 0.003630 | -0.112677 | 0.122711 |
| global-macro-production-5m | 5m | 1 | US industrial production index change | 12 monthly observation(s) | 0.062281 | 0.118437 | -0.085393 | -0.048468 |

## Best marginal candidate by screen

| screen | candidates | best candidate | stable | lookback | first half | second half | transfer |
|---|---:|---|:---:|---|---:|---:|---:|
| cross-market-1m | 104 | ETH realized volatility | yes | 30m | 0.075462 | 0.063778 | 0.036094 |
| cross-market-5m | 104 | ETH realized volatility | yes | 60m | 0.037844 | 0.016449 | 0.017613 |
| cross-market-15m | 104 | ETH realized volatility | yes | 60m | 0.014857 | 0.006360 | 0.005732 |
| cross-market-30m | 104 | SOL realized volatility | no | 5m | -0.000645 | 0.000014 | 0.000087 |
| cross-market-60m | 104 | ETH/BTC volatility ratio | no | 5m | -0.001630 | -0.002801 | -0.002705 |
| deribit-dvol-5m | 17 | DVOL change | no | 4h | -0.022652 | -0.026518 | -0.027324 |
| deribit-dvol-15m | 17 | Absolute DVOL change | no | 4h | -0.015782 | -0.034885 | -0.043760 |
| deribit-dvol-30m | 17 | BTC DVOL level | no | latest completed 1h | -0.023896 | -0.026771 | -0.027649 |
| deribit-dvol-60m | 17 | Absolute DVOL change | no | 168h | -0.031963 | -0.044529 | -0.035349 |
| cboe-vix-1m | 9 | VIX minus BTC realized volatility | no | 7d BTC realized | -0.006488 | -0.035174 | -0.001802 |
| cboe-vix-5m | 9 | VIX change | no | 21 trading day(s) | -0.040192 | -0.029401 | -0.040797 |
| cboe-vix-15m | 9 | VIX change | no | 5 trading day(s) | -0.021542 | -0.045237 | -0.019594 |
| cboe-vix-30m | 9 | VIX minus BTC realized volatility | no | 7d BTC realized | -0.015868 | -0.034663 | -0.027522 |
| cboe-vix-60m | 9 | VIX change | no | 21 trading day(s) | -0.033780 | -0.044182 | -0.040747 |
| global-macro-1m | 309 | India short-term interest rate change | no | 3 monthly observation(s) | 0.015919 | -0.008916 | 0.060182 |
| global-macro-5m | 309 | US 2-year Treasury yield absolute change | yes | 21 trading observation(s) | 0.030066 | 0.187949 | 0.111357 |
| global-macro-15m | 309 | United Kingdom industrial production index change | yes | 3 monthly observation(s) | 0.098798 | 0.168376 | 0.038805 |
| global-macro-30m | 309 | US 10-year minus 2-year yield spread change | yes | 1 trading observation(s) | 0.133043 | 0.167182 | 0.132182 |
| global-macro-60m | 309 | Euro-area AAA 2-year yield change | yes | 1 trading observation(s) | 0.107125 | 0.276729 | 0.015992 |
| binance-funding-1m | 18 | Mean settled funding rate | no | 9 settlement(s), normally 3.0d | -0.043520 | -0.106328 | -0.155133 |
| binance-funding-5m | 18 | Mean absolute settled funding rate | no | 90 settlement(s), normally 30.0d | -0.142236 | -0.150162 | -0.118974 |
| binance-funding-15m | 18 | Funding-rate change | no | 3 settlement(s), normally 1.0d | -0.067004 | -0.135692 | -0.038490 |
| binance-funding-30m | 18 | Mean settled funding rate | no | 90 settlement(s), normally 30.0d | -0.080619 | -0.102675 | -0.110710 |
| binance-funding-60m | 18 | BTCUSDT settled funding rate | no | latest settlement | -0.142918 | -0.096607 | -0.115793 |
| coinmetrics-15m | 37 | Total-fee change | no | 1d | -0.031710 | -0.024359 | -0.032310 |
| coinmetrics-30m | 37 | Hash-rate change | no | 7d | -0.042535 | -0.043981 | -0.037322 |
| coinmetrics-60m | 37 | Exchange inflow | no | 3d sum | -0.040880 | -0.035758 | -0.039153 |
| community-daily-15m | 145 | exchange stablecoins ratio usd change | no | 3d | -0.025313 | -0.020349 | -0.024508 |
| community-daily-30m | 145 | funding rates change | no | 30d | -0.025050 | -0.038036 | -0.026301 |
| community-daily-60m | 145 | exchange stablecoins ratio usd change | no | 1d | -0.036763 | -0.036480 | -0.037021 |
| mempool-proxy-15m | 36 | Median block fee rate change | no | 1 buckets | -0.031626 | -0.008288 | -0.028019 |
| mempool-proxy-30m | 36 | 90th-percentile block fee rate change | no | 2 buckets | -0.015790 | -0.029368 | -0.022585 |
| mempool-proxy-60m | 36 | Average block weight change | no | 6 buckets | -0.039481 | -0.037081 | -0.037786 |
| global-macro-production-1m | 309 | US 10-year Treasury yield change | yes | 21 trading observation(s) | 0.076705 | 0.121317 | 0.147175 |
| global-macro-production-5m | 309 | US industrial production index change | yes | 12 monthly observation(s) | 0.054992 | 0.069334 | 0.118437 |
| global-macro-production-15m | 309 | US 2-year Treasury yield absolute change | no | 1 trading observation(s) | -0.039658 | 0.002439 | -0.034013 |
| global-macro-production-30m | 309 | US nonfarm payroll employment change | no | 12 monthly observation(s) | -0.029589 | 0.003053 | 0.019711 |
| global-macro-production-60m | 309 | India short-term interest rate change | no | 3 monthly observation(s) | 0.029982 | -0.014517 | -0.011189 |
| global-macro-long-1m | 309 | US 2-year Treasury yield change | no | 1 trading observation(s) | -0.014947 | -0.022737 | -0.015410 |
| global-macro-long-5m | 309 | US 2-year Treasury yield | no | latest conservatively available release | -0.059746 | 0.012678 | -0.036189 |
| global-macro-long-15m | 309 | India CPI year-over-year change | no | 1 monthly observation(s) | -0.041056 | -0.029541 | -0.037175 |
| global-macro-long-30m | 309 | India short-term interest rate change | no | 12 monthly observation(s) | -0.062105 | -0.008472 | -0.006515 |
| global-macro-long-60m | 309 | Japan industrial production index absolute change | no | 3 monthly observation(s) | -0.025960 | -0.048464 | 0.053771 |
| binance-funding-long-1m | 18 | Funding-rate change | no | 21 settlement(s), normally 7.0d | -0.040061 | -0.020865 | -0.001115 |
| binance-funding-long-5m | 18 | Mean absolute settled funding rate | no | 90 settlement(s), normally 30.0d | -0.056897 | -0.043884 | -0.005802 |
| binance-funding-long-15m | 18 | Mean absolute settled funding rate | no | 90 settlement(s), normally 30.0d | -0.057225 | -0.062339 | -0.029146 |
| binance-funding-long-30m | 18 | Mean settled funding rate | no | 9 settlement(s), normally 3.0d | -0.071753 | -0.050042 | -0.007417 |
| binance-funding-long-60m | 18 | Mean absolute settled funding rate | no | 21 settlement(s), normally 7.0d | -0.019481 | -0.028014 | -0.020095 |
| joint-external-60m | 675 | Bank of Russia key rate change | no | 1 policy decision(s) | -0.018892 | -0.019447 | -0.022230 |

## Macro coverage and effective updates

The release count is more relevant than the number of target candles that inherit a value. `value changes` is even stricter for policy rates and other series that often repeat unchanged.

| series | economy | provider | frequency | assumed lag | rows | train releases | train value changes | primary test | transfer |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| BAMLH0A0HYM2: US high-yield credit spread | United States | Federal Reserve Bank of St. Louis FRED | daily | 1d | 786 | 131 | 117 | 43 | 47 |
| BOE_BANK_RATE: Bank of England Bank Rate | United Kingdom | Bank of England | event | 1d | 22 | 2 | 1 | 0 | 0 |
| CBR_CPI_YOY: Russia inflation year-over-year | Russia | Bank of Russia | monthly | 45d | 77 | 6 | 5 | 2 | 2 |
| CBR_KEY_RATE: Bank of Russia key rate | Russia | Bank of Russia | event | 1d | 37 | 2 | 1 | 2 | 2 |
| CLVMNACSCAB1GQEA19: Euro-area real GDP | Euro area | Federal Reserve Bank of St. Louis FRED | quarterly | 150d | 25 | 2 | 1 | 0 | 1 |
| CP0000EZ19M086NEST: Euro-area consumer price index | Euro area | Federal Reserve Bank of St. Louis FRED | monthly | 45d | 77 | 6 | 5 | 2 | 2 |
| CPIAUCSL: US consumer price index | United States | Federal Reserve Bank of St. Louis FRED | monthly | 45d | 76 | 6 | 5 | 2 | 2 |
| DEXCHUS: Chinese yuan per US dollar | China | Federal Reserve Bank of St. Louis FRED | daily | 1d | 1624 | 126 | 122 | 41 | 44 |
| DFF: US effective federal funds rate | United States | Federal Reserve Bank of St. Louis FRED | daily | 1d | 2373 | 180 | 0 | 61 | 64 |
| DGS10: US 10-year Treasury yield | United States | Federal Reserve Bank of St. Louis FRED | daily | 1d | 1626 | 125 | 117 | 41 | 44 |
| DGS2: US 2-year Treasury yield | United States | Federal Reserve Bank of St. Louis FRED | daily | 1d | 1626 | 125 | 105 | 41 | 44 |
| DTWEXBGS: Trade-weighted US dollar index | United States | Federal Reserve Bank of St. Louis FRED | daily | 1d | 1624 | 126 | 125 | 41 | 44 |
| ECBDFR: ECB deposit facility rate | Euro area | Federal Reserve Bank of St. Louis FRED | daily | 1d | 2375 | 180 | 2 | 61 | 64 |
| ECB_ESTR: Euro short-term rate (€STR) | Euro area | European Central Bank Data Portal | daily | 1d | 1665 | 127 | 99 | 43 | 45 |
| ECB_YC_10Y: Euro-area AAA 10-year yield | Euro area | European Central Bank Data Portal | daily | 1d | 1661 | 127 | 126 | 43 | 45 |
| ECB_YC_10Y2Y: Euro-area AAA 10-year minus 2-year yield spread | Euro area | European Central Bank Data Portal | daily | 1d | 1661 | 127 | 126 | 43 | 45 |
| ECB_YC_2Y: Euro-area AAA 2-year yield | Euro area | European Central Bank Data Portal | daily | 1d | 1661 | 127 | 126 | 43 | 45 |
| GDPC1: US real GDP | United States | Federal Reserve Bank of St. Louis FRED | quarterly | 150d | 25 | 2 | 1 | 0 | 1 |
| INDPRO: US industrial production index | United States | Federal Reserve Bank of St. Louis FRED | monthly | 50d | 77 | 6 | 5 | 2 | 2 |
| IRSTCI01INM156N: India short-term interest rate | India | Federal Reserve Bank of St. Louis FRED | monthly | 45d | 76 | 6 | 3 | 2 | 2 |
| IRSTCI01JPM156N: Japan short-term interest rate | Japan | Federal Reserve Bank of St. Louis FRED | monthly | 45d | 76 | 6 | 2 | 2 | 2 |
| JPNRGDPEXP: Japan real GDP | Japan | Federal Reserve Bank of St. Louis FRED | quarterly | 150d | 25 | 2 | 1 | 0 | 1 |
| OECD_CPI_YOY_CHN: China CPI year-over-year | China | OECD Data Explorer SDMX API | monthly | 45d | 76 | 6 | 3 | 2 | 2 |
| OECD_CPI_YOY_GBR: United Kingdom CPI year-over-year | United Kingdom | OECD Data Explorer SDMX API | monthly | 45d | 76 | 6 | 5 | 2 | 2 |
| OECD_CPI_YOY_IND: India CPI year-over-year | India | OECD Data Explorer SDMX API | monthly | 45d | 76 | 6 | 5 | 2 | 2 |
| OECD_CPI_YOY_JPN: Japan CPI year-over-year | Japan | OECD Data Explorer SDMX API | monthly | 45d | 76 | 6 | 4 | 2 | 2 |
| OECD_INDUSTRIAL_PRODUCTION_EA20: Euro area industrial production index | Euro area | OECD Data Explorer SDMX API | monthly | 50d | 75 | 6 | 4 | 2 | 2 |
| OECD_INDUSTRIAL_PRODUCTION_GBR: United Kingdom industrial production index | United Kingdom | OECD Data Explorer SDMX API | monthly | 50d | 75 | 6 | 5 | 2 | 2 |
| OECD_INDUSTRIAL_PRODUCTION_IND: India industrial production index | India | OECD Data Explorer SDMX API | monthly | 50d | 75 | 6 | 5 | 2 | 2 |
| OECD_INDUSTRIAL_PRODUCTION_JPN: Japan industrial production index | Japan | OECD Data Explorer SDMX API | monthly | 50d | 75 | 6 | 5 | 2 | 2 |
| OECD_REAL_GDP_QOQ_CHN: China real GDP quarter-over-quarter | China | OECD Data Explorer SDMX API | quarterly | 150d | 25 | 2 | 1 | 0 | 1 |
| OECD_REAL_GDP_QOQ_GBR: United Kingdom real GDP quarter-over-quarter | United Kingdom | OECD Data Explorer SDMX API | quarterly | 150d | 25 | 2 | 1 | 0 | 1 |
| OECD_REAL_GDP_QOQ_IND: India real GDP quarter-over-quarter | India | OECD Data Explorer SDMX API | quarterly | 150d | 24 | 2 | 1 | 0 | 1 |
| PAYEMS: US nonfarm payroll employment | United States | Federal Reserve Bank of St. Louis FRED | monthly | 40d | 77 | 6 | 5 | 2 | 2 |
| T10Y2Y: US 10-year minus 2-year yield spread | United States | Federal Reserve Bank of St. Louis FRED | daily | 1d | 1627 | 125 | 105 | 41 | 44 |
| UNRATE: US unemployment rate | United States | Federal Reserve Bank of St. Louis FRED | monthly | 40d | 76 | 6 | 3 | 1 | 2 |

## Interpretation constraints

- DVOL is an hourly volatility-index history, not a historical full option surface; ATM, 25-delta skew, term structure, strike OI, and IV/skew changes remain live-forward measurements.
- VIX is a daily US-equity option-implied volatility index. The backtest delays each close until the next UTC day and therefore tests it as a slow macro regime feature, not a live intraday VIX feed.
- Macro values are current revised observations, not point-in-time vintages. Conservative fixed publication lags prevent obvious same-period leakage, but any selected macro result remains provisional until repeated on vintage release data.
- The global macro discovery screen tests 309 correlated transformations. The three-block rule reduces but does not eliminate multiple-testing bias; the failed 2021-2026 screen overrides recent-window discoveries for production selection.
- Monthly CPI/labor/production and quarterly GDP have too few independent releases in a 180-day fit to establish short-horizon value. Daily sampling limits duplication, and per-series update counts are recorded, but these slow levels should be treated as regime metadata rather than proven candle predictors.
- Binance funding features use only the last settled BTCUSDT USD-M funding rates, delayed by one minute. Historical pre-settlement predicted funding and authenticated margin borrow rates are not present in the public endpoint.
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
- Slow sources: Daily evaluation for macro state, one evaluation per normal 8h settlement for funding, and hourly evaluation for other slow sources.

Complete candidate rankings, frozen quantile edges, observation counts, sign decomposition, and magnitude decomposition are in `data/benchmarks/public-external-feature-information.json`.
