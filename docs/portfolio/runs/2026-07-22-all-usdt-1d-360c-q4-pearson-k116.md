# Binance portfolio basis

Generated 2026-07-23T19:08:11.258Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2025-07-28 through 2026-07-22
- Sampling: exactly 360 1d log returns (360 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 1.0% of the maximum unexplained variance, select the largest mean absolute 1d return
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 408 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 162.11 bp | 100.0% | 0 | 1.4B | 43.5% |
| 2 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 100.0% | 0.016 | BTCUSDT | 981.44 bp | 100.0% | 0 | 6.6M | 415.9% |
| 3 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 100.0% | 0.023 | BTCUSDT | 837.90 bp | 100.0% | 24 | 2.7M | 387.2% |
| 4 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 99.7% | 0.069 | SIRENUSDT | 712.35 bp | 100.0% | 0 | 3M | 274.9% |
| 5 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 99.8% | 0.040 | BTCUSDT | 668.86 bp | 100.0% | 0 | 7.6M | 260.5% |
| 6 | TA | TAUSDT (usdm-futures) | usdm-futures | 99.0% | 0.110 | BTCUSDT | 611.85 bp | 100.0% | 0 | 4.5M | 203.6% |
| 7 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 99.4% | 0.091 | BTCUSDT | 482.82 bp | 100.0% | 24 | 3.7M | 152.1% |
| 8 | JST | JSTUSDT (spot) | spot, usdm-futures | 99.1% | 0.090 | BTCUSDT | 214.72 bp | 100.0% | 24 | 2.2M | 61.4% |
| 9 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 98.0% | 0.150 | BTCUSDT | 809.84 bp | 100.0% | 0 | 4.9M | 254.9% |
| 10 | TAC | TACUSDT (usdm-futures) | usdm-futures | 98.1% | 0.138 | BTCUSDT | 649.05 bp | 100.0% | 24 | 1.8M | 282.6% |
| 11 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 97.7% | 0.151 | BTCUSDT | 887.95 bp | 100.0% | 24 | 25.2M | 294.0% |
| 12 | BR | BRUSDT (usdm-futures) | usdm-futures | 97.8% | 0.121 | BTCUSDT | 565.08 bp | 100.0% | 48 | 1.5M | 192.5% |
| 13 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 97.4% | 0.166 | BTCUSDT | 515.37 bp | 100.0% | 24 | 1.4M | 227.1% |
| 14 | B2 | B2USDT (usdm-futures) | usdm-futures | 97.5% | 0.205 | BTCUSDT | 502.67 bp | 100.0% | 24 | 2.1M | 163.8% |
| 15 | M | MUSDT (usdm-futures) | usdm-futures | 97.2% | 0.140 | TACUSDT | 578.42 bp | 100.0% | 0 | 11.5M | 195.3% |
| 16 | OG | OGUSDT (spot) | spot, usdm-futures | 97.5% | 0.183 | BTCUSDT | 404.46 bp | 100.0% | 24 | 2.1M | 131.2% |
| 17 | H | HUSDT (usdm-futures) | usdm-futures | 96.6% | 0.155 | BTCUSDT | 961.49 bp | 100.0% | 0 | 25M | 371.5% |
| 18 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 96.6% | 0.132 | TACUSDT | 735.94 bp | 100.0% | 0 | 3.4M | 222.5% |
| 19 | BAN | BANUSDT (usdm-futures) | usdm-futures | 96.8% | 0.183 | BTCUSDT | 496.99 bp | 100.0% | 0 | 3.1M | 170.4% |
| 20 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 95.9% | 0.148 | HUSDT | 528.62 bp | 100.0% | 24 | 1.9M | 149.8% |
| 21 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 96.2% | 0.188 | BTCUSDT | 489.31 bp | 100.0% | 24 | 1M | 205.7% |
| 22 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 95.4% | 0.179 | BTCUSDT | 750.54 bp | 100.0% | 24 | 4.3M | 229.6% |
| 23 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 95.2% | 0.238 | BTCUSDT | 538.80 bp | 100.0% | 24 | 2.9M | 159.9% |
| 24 | STO | STOUSDT (spot) | spot, usdm-futures | 94.8% | 0.189 | BTCUSDT | 486.48 bp | 100.0% | 24 | 1.2M | 178.0% |
| 25 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 94.8% | 0.164 | BTCUSDT | 433.75 bp | 100.0% | 24 | 1.5M | 136.7% |
| 26 | B | BUSDT (usdm-futures) | usdm-futures | 93.9% | 0.276 | BTCUSDT | 670.52 bp | 100.0% | 0 | 4.8M | 214.2% |
| 27 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 92.9% | 0.224 | BTCUSDT | 582.90 bp | 100.0% | 24 | 3.6M | 152.6% |
| 28 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 93.3% | 0.188 | BTCUSDT | 531.22 bp | 100.0% | 48 | 2.3M | 153.7% |
| 29 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 92.0% | 0.267 | PIPPINUSDT | 695.63 bp | 100.0% | 0 | 9.2M | 205.8% |
| 30 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 92.2% | 0.233 | PIPPINUSDT | 643.58 bp | 100.0% | 24 | 4.7M | 216.4% |
| 31 | MERL | MERLUSDT (usdm-futures) | usdm-futures | 92.1% | 0.235 | BTCUSDT | 577.56 bp | 100.0% | 24 | 8.9M | 177.8% |
| 32 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 92.0% | 0.272 | BTCUSDT | 564.50 bp | 100.0% | 0 | 7.4M | 188.4% |
| 33 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 91.7% | 0.302 | SQDUSDT | 391.93 bp | 100.0% | 24 | 706K | 190.9% |
| 34 | PROM | PROMUSDT (spot) | spot, usdm-futures | 91.7% | 0.240 | DEXEUSDT | 370.85 bp | 100.0% | 24 | 1.2M | 121.8% |
| 35 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 90.9% | 0.205 | BTCUSDT | 911.37 bp | 100.0% | 24 | 18.6M | 319.4% |
| 36 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 90.4% | 0.216 | TACUSDT | 801.52 bp | 100.0% | 24 | 4.4M | 256.7% |
| 37 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 90.5% | 0.294 | BTCUSDT | 473.24 bp | 100.0% | 0 | 1.1M | 147.9% |
| 38 | DCR | DCRUSDT (spot) | spot | 89.8% | 0.333 | BTCUSDT | 398.60 bp | 100.0% | 24 | 438.1K | 120.5% |
| 39 | AWE | AWEUSDT (spot) | spot, usdm-futures | 89.6% | 0.328 | BTCUSDT | 326.61 bp | 100.0% | 24 | 475.5K | 95.1% |
| 40 | ATM | ATMUSDT (spot) | spot | 89.7% | 0.225 | DEXEUSDT | 313.98 bp | 100.0% | 24 | 692.8K | 101.2% |
| 41 | AIN | AINUSDT (usdm-futures) | usdm-futures | 88.2% | 0.229 | SKYAIUSDT | 572.19 bp | 100.0% | 0 | 2.1M | 169.1% |
| 42 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 88.0% | 0.223 | BTCUSDT | 526.88 bp | 100.0% | 24 | 1.7M | 161.4% |
| 43 | SUN | SUNUSDT (spot) | spot, usdm-futures | 88.4% | 0.268 | BTCUSDT | 190.60 bp | 100.0% | 24 | 1.1M | 63.6% |
| 44 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 87.6% | 0.300 | BTCUSDT | 387.98 bp | 100.0% | 24 | 2.3M | 129.7% |
| 45 | PAXG | PAXGUSDT (spot) | spot, usdm-futures | 87.0% | 0.332 | BTCUSDT | 97.53 bp | 100.0% | 0 | 24.1M | 29.1% |
| 46 | GUN | GUNUSDT (spot) | spot, usdm-futures | 85.9% | 0.321 | BTCUSDT | 527.02 bp | 100.0% | 24 | 1.6M | 146.2% |
| 47 | NIL | NILUSDT (spot) | spot, usdm-futures | 85.4% | 0.386 | BTCUSDT | 503.66 bp | 100.0% | 24 | 1.2M | 160.4% |
| 48 | STG | STGUSDT (spot) | spot, usdm-futures | 85.3% | 0.290 | BTCUSDT | 462.87 bp | 100.0% | 24 | 946.7K | 150.9% |
| 49 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 85.7% | 0.365 | BTCUSDT | 452.92 bp | 100.0% | 0 | 2M | 121.5% |
| 50 | XNO | XNOUSDT (spot) | spot | 85.0% | 0.320 | BTCUSDT | 322.24 bp | 100.0% | 48 | 212.8K | 105.0% |
| 51 | ZORA | ZORAUSDT (usdm-futures) | usdm-futures | 84.2% | 0.389 | BTCUSDT | 460.87 bp | 100.0% | 24 | 6.7M | 126.5% |
| 52 | SYN | SYNUSDT (spot) | spot, usdm-futures | 83.5% | 0.264 | BULLAUSDT | 513.77 bp | 100.0% | 24 | 554.5K | 169.9% |
| 53 | DUSK | DUSKUSDT (spot) | spot, usdm-futures | 83.4% | 0.305 | BTCUSDT | 502.13 bp | 100.0% | 24 | 864.6K | 146.5% |
| 54 | HEI | HEIUSDT (spot) | spot, usdm-futures | 83.5% | 0.236 | TAUSDT | 457.19 bp | 100.0% | 24 | 998.4K | 160.9% |
| 55 | LUNC | LUNCUSDT (spot) | spot, usdm-futures | 83.3% | 0.376 | BTCUSDT | 371.03 bp | 100.0% | 0 | 2.6M | 119.1% |
| 56 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 82.1% | 0.344 | BTCUSDT | 533.77 bp | 100.0% | 24 | 1.2M | 181.9% |
| 57 | GPS | GPSUSDT (spot) | spot, usdm-futures | 82.2% | 0.331 | PARTIUSDT | 466.55 bp | 100.0% | 24 | 1.1M | 140.6% |
| 58 | SIGN | SIGNUSDT (spot) | spot, usdm-futures | 82.2% | 0.300 | BTCUSDT | 360.09 bp | 100.0% | 48 | 1M | 115.3% |
| 59 | REQ | REQUSDT (spot) | spot | 82.1% | 0.425 | BTCUSDT | 212.10 bp | 100.0% | 24 | 190.7K | 69.6% |
| 60 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 81.4% | 0.312 | BTCUSDT | 473.21 bp | 100.0% | 0 | 2.2M | 149.8% |
| 61 | VVV | VVVUSDT (usdm-futures) | usdm-futures | 80.9% | 0.439 | BTCUSDT | 539.14 bp | 100.0% | 24 | 12.7M | 147.1% |
| 62 | RIF | RIFUSDT (spot) | spot, usdm-futures | 80.8% | 0.352 | DEXEUSDT | 443.35 bp | 100.0% | 24 | 533.7K | 139.8% |
| 63 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 80.3% | 0.314 | TAUSDT | 536.94 bp | 100.0% | 24 | 3.6M | 167.9% |
| 64 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 80.5% | 0.319 | BTCUSDT | 421.42 bp | 100.0% | 0 | 1.6M | 148.2% |
| 65 | NMR | NMRUSDT (spot) | spot, usdm-futures | 80.3% | 0.351 | TNSRUSDT | 351.46 bp | 100.0% | 24 | 892.1K | 123.9% |
| 66 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 79.2% | 0.334 | HUSDT | 536.94 bp | 100.0% | 24 | 1.4M | 163.0% |
| 67 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 79.1% | 0.312 | BTCUSDT | 534.82 bp | 100.0% | 24 | 2.5M | 162.7% |
| 68 | DRIFT | DRIFTUSDT (usdm-futures) | usdm-futures | 78.9% | 0.426 | BTCUSDT | 512.56 bp | 100.0% | 24 | 3.6M | 139.6% |
| 69 | BICO | BICOUSDT (spot) | spot, usdm-futures | 78.3% | 0.438 | BTCUSDT | 412.67 bp | 100.0% | 24 | 416.7K | 136.1% |
| 70 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 78.0% | 0.405 | BROCCOLIF3BUSDT | 395.32 bp | 100.0% | 24 | 844.6K | 236.3% |
| 71 | SKL | SKLUSDT (spot) | spot, usdm-futures | 77.3% | 0.463 | BTCUSDT | 379.02 bp | 100.0% | 24 | 788.7K | 117.5% |
| 72 | ASR | ASRUSDT (spot) | spot, usdm-futures | 77.1% | 0.451 | ATMUSDT | 305.76 bp | 100.0% | 24 | 755.1K | 103.2% |
| 73 | PIVX | PIVXUSDT (spot) | spot | 76.1% | 0.308 | BTCUSDT | 445.64 bp | 100.0% | 48 | 337.8K | 131.5% |
| 74 | SPK | SPKUSDT (spot) | spot, usdm-futures | 76.3% | 0.440 | BTCUSDT | 384.20 bp | 100.0% | 0 | 2.4M | 109.3% |
| 75 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 75.8% | 0.442 | BTCUSDT | 437.45 bp | 100.0% | 24 | 1.3M | 160.0% |
| 76 | BABY | BABYUSDT (spot) | spot, usdm-futures | 75.5% | 0.424 | BTCUSDT | 402.06 bp | 100.0% | 24 | 1.2M | 115.6% |
| 77 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 75.5% | 0.440 | BTCUSDT | 388.54 bp | 100.0% | 24 | 525.7K | 121.3% |
| 78 | FLOW | FLOWUSDT (spot) | spot, usdm-futures | 75.5% | 0.480 | BTCUSDT | 364.93 bp | 100.0% | 24 | 972.5K | 112.0% |
| 79 | LSK | LSKUSDT (spot) | spot, usdm-futures | 75.0% | 0.468 | BTCUSDT | 327.05 bp | 100.0% | 48 | 347.2K | 102.5% |
| 80 | PORTO | PORTOUSDT (spot) | spot | 74.7% | 0.492 | BTCUSDT | 278.27 bp | 100.0% | 24 | 291.9K | 87.3% |
| 81 | TST | TSTUSDT (spot) | spot, usdm-futures | 73.8% | 0.340 | BTCUSDT | 514.93 bp | 100.0% | 0 | 1.3M | 164.6% |
| 82 | C | CUSDT (spot) | spot, usdm-futures | 74.1% | 0.349 | MAVIAUSDT | 467.82 bp | 100.0% | 24 | 1.2M | 131.7% |
| 83 | ORCA | ORCAUSDT (spot) | spot, usdm-futures | 73.7% | 0.452 | BTCUSDT | 368.91 bp | 100.0% | 24 | 812.5K | 108.9% |
| 84 | HYPE | HYPEUSDT (usdm-futures) | usdm-futures | 73.4% | 0.504 | BTCUSDT | 389.16 bp | 100.0% | 0 | 408.3M | 96.1% |
| 85 | STRAX | STRAXUSDT (spot) | spot | 73.6% | 0.437 | BTCUSDT | 278.76 bp | 100.0% | 24 | 374.1K | 93.5% |
| 86 | EDU | EDUUSDT (spot) | spot, usdm-futures | 73.2% | 0.408 | BTCUSDT | 395.52 bp | 100.0% | 24 | 1.1M | 124.7% |
| 87 | WIN | WINUSDT (spot) | spot | 73.4% | 0.429 | BTCUSDT | 239.25 bp | 100.0% | 24 | 307.3K | 79.5% |
| 88 | PYR | PYRUSDT (spot) | spot | 72.8% | 0.401 | XNOUSDT | 368.46 bp | 100.0% | 48 | 1M | 123.7% |
| 89 | API3 | API3USDT (spot) | spot, usdm-futures | 72.8% | 0.493 | BTCUSDT | 340.05 bp | 100.0% | 0 | 960K | 98.0% |
| 90 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 71.9% | 0.341 | BANANAS31USDT | 633.23 bp | 100.0% | 24 | 2.6M | 182.3% |
| 91 | ONT | ONTUSDT (spot) | spot, usdm-futures | 71.8% | 0.440 | BTCUSDT | 350.19 bp | 100.0% | 24 | 623.5K | 109.0% |
| 92 | G | GUSDT (spot) | spot, usdm-futures | 71.9% | 0.473 | BTCUSDT | 333.08 bp | 100.0% | 24 | 393.8K | 102.5% |
| 93 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 71.4% | 0.485 | BTCUSDT | 408.44 bp | 100.0% | 24 | 1.2M | 122.6% |
| 94 | ENJ | ENJUSDT (spot) | spot, usdm-futures | 71.3% | 0.407 | BTCUSDT | 371.14 bp | 100.0% | 24 | 830.7K | 114.3% |
| 95 | OSMO | OSMOUSDT (spot) | spot | 70.9% | 0.445 | BTCUSDT | 357.28 bp | 100.0% | 24 | 312.7K | 122.1% |
| 96 | TWT | TWTUSDT (spot) | spot, usdm-futures | 70.6% | 0.446 | BTCUSDT | 278.85 bp | 100.0% | 24 | 1.1M | 81.4% |
| 97 | CHEEMS | 1000CHEEMSUSDT (spot) | spot, usdm-futures | 70.1% | 0.505 | BTCUSDT | 390.98 bp | 100.0% | 24 | 1M | 106.1% |
| 98 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 70.2% | 0.462 | BTCUSDT | 299.61 bp | 100.0% | 24 | 532.4K | 98.2% |
| 99 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 70.2% | 0.444 | BTCUSDT | 271.24 bp | 100.0% | 24 | 1.3M | 81.5% |
| 100 | DGB | DGBUSDT (spot) | spot | 69.6% | 0.483 | BTCUSDT | 329.53 bp | 100.0% | 24 | 321.2K | 92.6% |
| 101 | ZEC | ZECUSDT (spot) | spot, usdm-futures | 68.6% | 0.433 | BTCUSDT | 564.30 bp | 100.0% | 24 | 89M | 152.9% |
| 102 | BIO | BIOUSDT (spot) | spot, usdm-futures | 68.4% | 0.450 | ENJUSDT | 535.90 bp | 100.0% | 24 | 4.8M | 156.7% |
| 103 | BERA | BERAUSDT (spot) | spot, usdm-futures | 68.7% | 0.484 | BTCUSDT | 449.86 bp | 100.0% | 48 | 2.9M | 135.9% |
| 104 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 68.1% | 0.496 | BTCUSDT | 301.57 bp | 100.0% | 0 | 32.5M | 84.7% |
| 105 | ACX | ACXUSDT (spot) | spot, usdm-futures | 68.2% | 0.456 | BTCUSDT | 296.59 bp | 100.0% | 24 | 285.9K | 99.5% |
| 106 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 67.5% | 0.410 | BTCUSDT | 584.99 bp | 100.0% | 0 | 4.3M | 165.7% |
| 107 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 67.5% | 0.423 | FIDAUSDT | 451.85 bp | 100.0% | 24 | 833.5K | 136.2% |
| 108 | RED | REDUSDT (spot) | spot, usdm-futures | 67.7% | 0.461 | BTCUSDT | 391.46 bp | 100.0% | 24 | 775.2K | 120.5% |
| 109 | LA | LAUSDT (spot) | spot, usdm-futures | 67.0% | 0.427 | BTCUSDT | 410.82 bp | 100.0% | 24 | 961.3K | 108.7% |
| 110 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 67.0% | 0.554 | BTCUSDT | 391.29 bp | 100.0% | 24 | 1.2M | 101.3% |
| 111 | VIC | VICUSDT (spot) | spot, usdm-futures | 66.5% | 0.434 | LUMIAUSDT | 380.69 bp | 100.0% | 24 | 437.8K | 121.2% |
| 112 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 66.1% | 0.461 | BTCUSDT | 406.30 bp | 100.0% | 0 | 2.3M | 103.4% |
| 113 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 65.5% | 0.484 | STRAXUSDT | 510.36 bp | 100.0% | 24 | 903.3K | 172.8% |
| 114 | ACT | ACTUSDT (spot) | spot, usdm-futures | 65.6% | 0.446 | SWARMSUSDT | 441.49 bp | 100.0% | 48 | 1.9M | 124.9% |
| 115 | MOVR | MOVRUSDT (spot) | spot, usdm-futures | 65.7% | 0.442 | BTCUSDT | 400.15 bp | 100.0% | 24 | 784.4K | 132.4% |
| 116 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 65.5% | 0.401 | BTCUSDT | 386.62 bp | 100.0% | 24 | 1.2M | 111.5% |

## Diagnostics

- Basis size selected: 116
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.169
- Maximum pairwise absolute correlation: 0.554
- Mean whole-market projection R²: 81.8%
- Median whole-market projection R²: 80.0%
- 10th-percentile whole-market projection R²: 64.6%
- Minimum whole-market projection R²: 57.1%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30.1% | 30.1% | 5.7% | 0.0% |
| 5 | 32.5% | 31.9% | 9.5% | 1.0% |
| 10 | 37.0% | 36.2% | 13.8% | 3.9% |
| 15 | 40.3% | 39.4% | 17.7% | 4.9% |
| 20 | 43.0% | 41.0% | 20.7% | 7.5% |
| 25 | 47.5% | 45.9% | 23.5% | 11.9% |
| 30 | 50.8% | 48.8% | 27.0% | 15.2% |
| 35 | 53.2% | 51.8% | 29.5% | 18.0% |
| 40 | 55.9% | 54.5% | 32.5% | 21.7% |
| 45 | 58.2% | 56.4% | 35.3% | 26.2% |
| 50 | 61.5% | 60.0% | 39.3% | 29.2% |
| 55 | 64.0% | 62.1% | 42.3% | 32.0% |
| 60 | 65.9% | 63.9% | 44.0% | 34.3% |
| 65 | 67.5% | 65.1% | 45.4% | 37.2% |
| 70 | 69.3% | 67.0% | 47.1% | 40.3% |
| 75 | 70.9% | 68.5% | 48.9% | 42.9% |
| 80 | 72.5% | 70.3% | 51.3% | 45.0% |
| 85 | 73.9% | 71.8% | 53.0% | 45.9% |
| 90 | 75.4% | 73.4% | 55.0% | 48.1% |
| 95 | 76.8% | 75.0% | 57.4% | 50.1% |
| 100 | 78.1% | 76.5% | 59.4% | 52.8% |
| 105 | 79.4% | 77.5% | 61.6% | 54.0% |
| 110 | 80.5% | 78.5% | 63.1% | 55.4% |
| 115 | 81.6% | 79.9% | 64.4% | 56.9% |
| 116 | 81.8% | 80.0% | 64.6% | 57.1% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | SIREN | BULLA | VELVET | JELLYJELLY | TA | ALCH | JST | FHE | TAC | PIPPIN | BR | DEXE | B2 | M | OG | H | AGT | BAN | IDOL | TNSR | AIOT | SQD | STO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.016 | 0.023 | 0.007 | 0.040 | 0.110 | 0.091 | 0.090 | 0.150 | 0.138 | 0.151 | 0.121 | 0.166 | 0.205 | 0.060 | 0.183 | 0.155 | 0.077 | 0.183 | 0.141 | 0.188 | 0.179 | 0.238 | 0.189 |
| SIREN | 0.016 | 1.000 | 0.013 | 0.069 | -0.001 | -0.022 | -0.022 | -0.005 | -0.013 | 0.042 | 0.070 | 0.031 | 0.069 | 0.020 | 0.128 | 0.002 | -0.020 | 0.102 | 0.039 | 0.083 | 0.033 | -0.105 | 0.019 | 0.018 |
| BULLA | 0.023 | 0.013 | 1.000 | 0.025 | 0.017 | 0.042 | -0.022 | 0.033 | 0.053 | 0.035 | 0.004 | 0.021 | 0.007 | 0.018 | -0.016 | -0.003 | 0.033 | 0.097 | 0.056 | 0.041 | 0.067 | -0.027 | 0.045 | 0.031 |
| VELVET | 0.007 | 0.069 | 0.025 | 1.000 | -0.036 | 0.030 | -0.054 | 0.024 | 0.063 | 0.063 | 0.037 | 0.029 | 0.066 | 0.039 | -0.073 | -0.055 | 0.026 | 0.095 | -0.016 | 0.115 | 0.006 | 0.090 | -0.072 | 0.049 |
| JELLYJELLY | 0.040 | -0.001 | 0.017 | -0.036 | 1.000 | -0.062 | 0.029 | 0.008 | 0.005 | 0.015 | 0.054 | 0.037 | -0.025 | 0.018 | 0.049 | 0.001 | 0.143 | -0.032 | 0.058 | 0.111 | -0.036 | -0.028 | -0.030 | 0.037 |
| TA | 0.110 | -0.022 | 0.042 | 0.030 | -0.062 | 1.000 | 0.019 | 0.062 | 0.106 | 0.100 | 0.105 | 0.080 | 0.128 | 0.061 | 0.019 | 0.043 | 0.026 | 0.131 | -0.078 | 0.056 | 0.094 | 0.068 | 0.107 | 0.022 |
| ALCH | 0.091 | -0.022 | -0.022 | -0.054 | 0.029 | 0.019 | 1.000 | -0.068 | 0.054 | 0.073 | 0.070 | 0.118 | -0.003 | 0.030 | 0.028 | 0.057 | 0.030 | 0.027 | 0.004 | -0.005 | -0.000 | 0.006 | 0.033 | -0.023 |
| JST | 0.090 | -0.005 | 0.033 | 0.024 | 0.008 | 0.062 | -0.068 | 1.000 | -0.026 | 0.045 | 0.017 | -0.031 | 0.049 | 0.028 | 0.040 | 0.009 | 0.025 | 0.039 | 0.072 | 0.044 | 0.013 | -0.006 | -0.009 | -0.057 |
| FHE | 0.150 | -0.013 | 0.053 | 0.063 | 0.005 | 0.106 | 0.054 | -0.026 | 1.000 | 0.040 | 0.031 | -0.036 | 0.046 | 0.033 | 0.008 | 0.042 | 0.050 | 0.058 | 0.038 | -0.006 | 0.028 | -0.002 | 0.076 | 0.056 |
| TAC | 0.138 | 0.042 | 0.035 | 0.063 | 0.015 | 0.100 | 0.073 | 0.045 | 0.040 | 1.000 | 0.086 | 0.060 | 0.028 | 0.032 | -0.140 | 0.007 | 0.115 | 0.132 | 0.025 | 0.043 | 0.071 | 0.098 | 0.055 | 0.066 |
| PIPPIN | 0.151 | 0.070 | 0.004 | 0.037 | 0.054 | 0.105 | 0.070 | 0.017 | 0.031 | 0.086 | 1.000 | 0.073 | 0.072 | 0.001 | 0.022 | 0.090 | -0.011 | 0.076 | 0.046 | -0.049 | 0.066 | 0.036 | 0.061 | 0.183 |
| BR | 0.121 | 0.031 | 0.021 | 0.029 | 0.037 | 0.080 | 0.118 | -0.031 | -0.036 | 0.060 | 0.073 | 1.000 | 0.057 | 0.072 | -0.008 | 0.070 | 0.093 | 0.017 | 0.070 | 0.007 | 0.052 | 0.118 | 0.060 | -0.006 |
| DEXE | 0.166 | 0.069 | 0.007 | 0.066 | -0.025 | 0.128 | -0.003 | 0.049 | 0.046 | 0.028 | 0.072 | 0.057 | 1.000 | 0.046 | 0.023 | 0.006 | 0.002 | 0.004 | 0.023 | 0.058 | 0.061 | 0.013 | 0.050 | 0.038 |
| B2 | 0.205 | 0.020 | 0.018 | 0.039 | 0.018 | 0.061 | 0.030 | 0.028 | 0.033 | 0.032 | 0.001 | 0.072 | 0.046 | 1.000 | -0.009 | -0.014 | 0.020 | 0.015 | 0.082 | 0.031 | 0.150 | 0.090 | -0.008 | 0.027 |
| M | 0.060 | 0.128 | -0.016 | -0.073 | 0.049 | 0.019 | 0.028 | 0.040 | 0.008 | -0.140 | 0.022 | -0.008 | 0.023 | -0.009 | 1.000 | 0.034 | 0.065 | -0.050 | 0.017 | -0.015 | 0.008 | 0.019 | 0.057 | -0.004 |
| OG | 0.183 | 0.002 | -0.003 | -0.055 | 0.001 | 0.043 | 0.057 | 0.009 | 0.042 | 0.007 | 0.090 | 0.070 | 0.006 | -0.014 | 0.034 | 1.000 | 0.024 | 0.052 | 0.074 | -0.002 | 0.041 | -0.030 | 0.099 | 0.127 |
| H | 0.155 | -0.020 | 0.033 | 0.026 | 0.143 | 0.026 | 0.030 | 0.025 | 0.050 | 0.115 | -0.011 | 0.093 | 0.002 | 0.020 | 0.065 | 0.024 | 1.000 | 0.055 | 0.063 | 0.148 | 0.061 | 0.001 | 0.088 | 0.032 |
| AGT | 0.077 | 0.102 | 0.097 | 0.095 | -0.032 | 0.131 | 0.027 | 0.039 | 0.058 | 0.132 | 0.076 | 0.017 | 0.004 | 0.015 | -0.050 | 0.052 | 0.055 | 1.000 | -0.021 | -0.010 | 0.133 | 0.033 | 0.062 | 0.088 |
| BAN | 0.183 | 0.039 | 0.056 | -0.016 | 0.058 | -0.078 | 0.004 | 0.072 | 0.038 | 0.025 | 0.046 | 0.070 | 0.023 | 0.082 | 0.017 | 0.074 | 0.063 | -0.021 | 1.000 | 0.051 | 0.021 | 0.060 | -0.004 | -0.006 |
| IDOL | 0.141 | 0.083 | 0.041 | 0.115 | 0.111 | 0.056 | -0.005 | 0.044 | -0.006 | 0.043 | -0.049 | 0.007 | 0.058 | 0.031 | -0.015 | -0.002 | 0.148 | -0.010 | 0.051 | 1.000 | 0.027 | 0.077 | 0.024 | -0.009 |
| TNSR | 0.188 | 0.033 | 0.067 | 0.006 | -0.036 | 0.094 | -0.000 | 0.013 | 0.028 | 0.071 | 0.066 | 0.052 | 0.061 | 0.150 | 0.008 | 0.041 | 0.061 | 0.133 | 0.021 | 0.027 | 1.000 | 0.041 | 0.039 | 0.059 |
| AIOT | 0.179 | -0.105 | -0.027 | 0.090 | -0.028 | 0.068 | 0.006 | -0.006 | -0.002 | 0.098 | 0.036 | 0.118 | 0.013 | 0.090 | 0.019 | -0.030 | 0.001 | 0.033 | 0.060 | 0.077 | 0.041 | 1.000 | -0.031 | -0.066 |
| SQD | 0.238 | 0.019 | 0.045 | -0.072 | -0.030 | 0.107 | 0.033 | -0.009 | 0.076 | 0.055 | 0.061 | 0.060 | 0.050 | -0.008 | 0.057 | 0.099 | 0.088 | 0.062 | -0.004 | 0.024 | 0.039 | -0.031 | 1.000 | 0.097 |
| STO | 0.189 | 0.018 | 0.031 | 0.049 | 0.037 | 0.022 | -0.023 | -0.057 | 0.056 | 0.066 | 0.183 | -0.006 | 0.038 | 0.027 | -0.004 | 0.127 | 0.032 | 0.088 | -0.006 | -0.009 | 0.059 | -0.066 | 0.097 | 1.000 |

The complete 116 × 116 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| TRX | TRXUSDT | 57.1% | 65.5% | SUNUSDT | 0.454 |
| QKC | QKCUSDT | 58.0% | 64.8% | BTCUSDT | 0.568 |
| FORM | FORMUSDT | 58.0% | 64.8% | BTCUSDT | 0.440 |
| TLM | TLMUSDT | 58.1% | 64.7% | BTCUSDT | 0.433 |
| CATI | CATIUSDT | 58.4% | 64.5% | BTCUSDT | 0.421 |
| BROCCOLI714 | BROCCOLI714USDT | 58.6% | 64.3% | KOMAUSDT | 0.495 |
| TUT | TUTUSDT | 58.7% | 64.3% | TSTUSDT | 0.470 |
| PHA | PHAUSDT | 58.9% | 64.1% | BTCUSDT | 0.488 |
| FTT | FTTUSDT | 59.0% | 64.0% | LUNCUSDT | 0.480 |
| QNT | QNTUSDT | 59.3% | 63.8% | BTCUSDT | 0.528 |
| KERNEL | KERNELUSDT | 59.6% | 63.6% | API3USDT | 0.494 |
| YFI | YFIUSDT | 59.8% | 63.4% | BTCUSDT | 0.574 |
| KAIA | KAIAUSDT | 60.1% | 63.2% | BTCUSDT | 0.589 |
| PYTH | PYTHUSDT | 60.6% | 62.8% | BTCUSDT | 0.569 |
| DOOD | DOODUSDT | 60.7% | 62.7% | BTCUSDT | 0.491 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

