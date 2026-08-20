# Binance portfolio basis

Generated 2026-07-23T19:16:57.605Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2025-07-28 through 2026-07-22
- Sampling: exactly 360 1d log returns (360 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 5.0% of the maximum unexplained variance, select the largest mean absolute 1d return
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
| 3 | H | HUSDT (usdm-futures) | usdm-futures | 98.8% | 0.155 | BTCUSDT | 961.49 bp | 100.0% | 0 | 25M | 371.5% |
| 4 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 98.6% | 0.151 | BTCUSDT | 887.95 bp | 100.0% | 24 | 25.2M | 294.0% |
| 5 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.033 | HUSDT | 837.90 bp | 100.0% | 24 | 2.7M | 387.2% |
| 6 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 98.7% | 0.150 | BTCUSDT | 809.84 bp | 100.0% | 0 | 4.9M | 254.9% |
| 7 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 96.9% | 0.202 | BTCUSDT | 801.52 bp | 100.0% | 24 | 4.4M | 256.7% |
| 8 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 98.2% | 0.102 | SIRENUSDT | 735.94 bp | 100.0% | 0 | 3.4M | 222.5% |
| 9 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 99.1% | 0.095 | AGTUSDT | 712.35 bp | 100.0% | 0 | 3M | 274.9% |
| 10 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 98.4% | 0.143 | HUSDT | 668.86 bp | 100.0% | 0 | 7.6M | 260.5% |
| 11 | M | MUSDT (usdm-futures) | usdm-futures | 98.1% | 0.128 | SIRENUSDT | 578.42 bp | 100.0% | 0 | 11.5M | 195.3% |
| 12 | BR | BRUSDT (usdm-futures) | usdm-futures | 98.2% | 0.121 | BTCUSDT | 565.08 bp | 100.0% | 48 | 1.5M | 192.5% |
| 13 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 97.6% | 0.166 | BTCUSDT | 515.37 bp | 100.0% | 24 | 1.4M | 227.1% |
| 14 | B2 | B2USDT (usdm-futures) | usdm-futures | 97.1% | 0.205 | BTCUSDT | 502.67 bp | 100.0% | 24 | 2.1M | 163.8% |
| 15 | BAN | BANUSDT (usdm-futures) | usdm-futures | 97.6% | 0.183 | BTCUSDT | 496.99 bp | 100.0% | 0 | 3.1M | 170.4% |
| 16 | STO | STOUSDT (spot) | spot, usdm-futures | 96.4% | 0.189 | BTCUSDT | 486.48 bp | 100.0% | 24 | 1.2M | 178.0% |
| 17 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 96.3% | 0.188 | BTCUSDT | 489.31 bp | 100.0% | 24 | 1M | 205.7% |
| 18 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 98.1% | 0.118 | BRUSDT | 482.82 bp | 100.0% | 24 | 3.7M | 152.1% |
| 19 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.2% | 0.164 | BTCUSDT | 433.75 bp | 100.0% | 24 | 1.5M | 136.7% |
| 20 | OG | OGUSDT (spot) | spot, usdm-futures | 96.8% | 0.183 | BTCUSDT | 404.46 bp | 100.0% | 24 | 2.1M | 131.2% |
| 21 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 95.9% | 0.158 | BTCUSDT | 391.93 bp | 100.0% | 24 | 706K | 190.9% |
| 22 | JST | JSTUSDT (spot) | spot, usdm-futures | 98.0% | 0.090 | BTCUSDT | 214.72 bp | 100.0% | 24 | 2.2M | 61.4% |
| 23 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 94.1% | 0.190 | SKYAIUSDT | 750.54 bp | 100.0% | 24 | 4.3M | 229.6% |
| 24 | B | BUSDT (usdm-futures) | usdm-futures | 94.2% | 0.276 | BTCUSDT | 670.52 bp | 100.0% | 0 | 4.8M | 214.2% |
| 25 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 92.5% | 0.205 | BTCUSDT | 911.37 bp | 100.0% | 24 | 18.6M | 319.4% |
| 26 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 92.6% | 0.267 | PIPPINUSDT | 695.63 bp | 100.0% | 0 | 9.2M | 205.8% |
| 27 | TAC | TACUSDT (usdm-futures) | usdm-futures | 93.7% | 0.216 | SKYAIUSDT | 649.05 bp | 100.0% | 24 | 1.8M | 282.6% |
| 28 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 92.4% | 0.233 | PIPPINUSDT | 643.58 bp | 100.0% | 24 | 4.7M | 216.4% |
| 29 | TA | TAUSDT (usdm-futures) | usdm-futures | 94.0% | 0.195 | SKYAIUSDT | 611.85 bp | 100.0% | 0 | 4.5M | 203.6% |
| 30 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 92.2% | 0.224 | BTCUSDT | 582.90 bp | 100.0% | 24 | 3.6M | 152.6% |
| 31 | MERL | MERLUSDT (usdm-futures) | usdm-futures | 92.7% | 0.235 | BTCUSDT | 577.56 bp | 100.0% | 24 | 8.9M | 177.8% |
| 32 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 92.3% | 0.272 | BTCUSDT | 564.50 bp | 100.0% | 0 | 7.4M | 188.4% |
| 33 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 91.7% | 0.208 | SKYAIUSDT | 531.22 bp | 100.0% | 48 | 2.3M | 153.7% |
| 34 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 93.8% | 0.148 | HUSDT | 528.62 bp | 100.0% | 24 | 1.9M | 149.8% |
| 35 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 90.5% | 0.302 | ALPINEUSDT | 538.80 bp | 100.0% | 24 | 2.9M | 159.9% |
| 36 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 90.5% | 0.294 | BTCUSDT | 473.24 bp | 100.0% | 0 | 1.1M | 147.9% |
| 37 | DCR | DCRUSDT (spot) | spot | 89.9% | 0.333 | BTCUSDT | 398.60 bp | 100.0% | 24 | 438.1K | 120.5% |
| 38 | PROM | PROMUSDT (spot) | spot, usdm-futures | 91.5% | 0.240 | DEXEUSDT | 370.85 bp | 100.0% | 24 | 1.2M | 121.8% |
| 39 | AIN | AINUSDT (usdm-futures) | usdm-futures | 89.1% | 0.229 | SKYAIUSDT | 572.19 bp | 100.0% | 0 | 2.1M | 169.1% |
| 40 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 88.7% | 0.223 | BTCUSDT | 526.88 bp | 100.0% | 24 | 1.7M | 161.4% |
| 41 | STG | STGUSDT (spot) | spot, usdm-futures | 87.0% | 0.290 | BTCUSDT | 462.87 bp | 100.0% | 24 | 946.7K | 150.9% |
| 42 | GUN | GUNUSDT (spot) | spot, usdm-futures | 86.6% | 0.321 | BTCUSDT | 527.02 bp | 100.0% | 24 | 1.6M | 146.2% |
| 43 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 87.2% | 0.300 | BTCUSDT | 387.98 bp | 100.0% | 24 | 2.3M | 129.7% |
| 44 | AWE | AWEUSDT (spot) | spot, usdm-futures | 88.6% | 0.328 | BTCUSDT | 326.61 bp | 100.0% | 24 | 475.5K | 95.1% |
| 45 | ATM | ATMUSDT (spot) | spot | 87.9% | 0.225 | DEXEUSDT | 313.98 bp | 100.0% | 24 | 692.8K | 101.2% |
| 46 | SUN | SUNUSDT (spot) | spot, usdm-futures | 88.2% | 0.268 | BTCUSDT | 190.60 bp | 100.0% | 24 | 1.1M | 63.6% |
| 47 | SYN | SYNUSDT (spot) | spot, usdm-futures | 84.8% | 0.264 | BULLAUSDT | 513.77 bp | 100.0% | 24 | 554.5K | 169.9% |
| 48 | NIL | NILUSDT (spot) | spot, usdm-futures | 84.6% | 0.386 | BTCUSDT | 503.66 bp | 100.0% | 24 | 1.2M | 160.4% |
| 49 | DUSK | DUSKUSDT (spot) | spot, usdm-futures | 84.8% | 0.305 | BTCUSDT | 502.13 bp | 100.0% | 24 | 864.6K | 146.5% |
| 50 | ZORA | ZORAUSDT (usdm-futures) | usdm-futures | 84.3% | 0.389 | BTCUSDT | 460.87 bp | 100.0% | 24 | 6.7M | 126.5% |
| 51 | HEI | HEIUSDT (spot) | spot, usdm-futures | 84.0% | 0.236 | TAUSDT | 457.19 bp | 100.0% | 24 | 998.4K | 160.9% |
| 52 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 85.1% | 0.365 | BTCUSDT | 452.92 bp | 100.0% | 0 | 2M | 121.5% |
| 53 | LUNC | LUNCUSDT (spot) | spot, usdm-futures | 83.6% | 0.376 | BTCUSDT | 371.03 bp | 100.0% | 0 | 2.6M | 119.1% |
| 54 | XNO | XNOUSDT (spot) | spot | 83.5% | 0.320 | BTCUSDT | 322.24 bp | 100.0% | 48 | 212.8K | 105.0% |
| 55 | PAXG | PAXGUSDT (spot) | spot, usdm-futures | 85.5% | 0.332 | BTCUSDT | 97.53 bp | 100.0% | 0 | 24.1M | 29.1% |
| 56 | VVV | VVVUSDT (usdm-futures) | usdm-futures | 81.2% | 0.439 | BTCUSDT | 539.14 bp | 100.0% | 24 | 12.7M | 147.1% |
| 57 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 80.5% | 0.314 | TAUSDT | 536.94 bp | 100.0% | 24 | 3.6M | 167.9% |
| 58 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 80.3% | 0.334 | HUSDT | 536.94 bp | 100.0% | 24 | 1.4M | 163.0% |
| 59 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 81.1% | 0.312 | BTCUSDT | 534.82 bp | 100.0% | 24 | 2.5M | 162.7% |
| 60 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 81.8% | 0.344 | BTCUSDT | 533.77 bp | 100.0% | 24 | 1.2M | 181.9% |
| 61 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 81.3% | 0.312 | BTCUSDT | 473.21 bp | 100.0% | 0 | 2.2M | 149.8% |
| 62 | GPS | GPSUSDT (spot) | spot, usdm-futures | 81.2% | 0.331 | PARTIUSDT | 466.55 bp | 100.0% | 24 | 1.1M | 140.6% |
| 63 | RIF | RIFUSDT (spot) | spot, usdm-futures | 80.3% | 0.352 | DEXEUSDT | 443.35 bp | 100.0% | 24 | 533.7K | 139.8% |
| 64 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 80.2% | 0.319 | BTCUSDT | 421.42 bp | 100.0% | 0 | 1.6M | 148.2% |
| 65 | SIGN | SIGNUSDT (spot) | spot, usdm-futures | 81.3% | 0.300 | BTCUSDT | 360.09 bp | 100.0% | 48 | 1M | 115.3% |
| 66 | NMR | NMRUSDT (spot) | spot, usdm-futures | 80.2% | 0.351 | TNSRUSDT | 351.46 bp | 100.0% | 24 | 892.1K | 123.9% |
| 67 | REQ | REQUSDT (spot) | spot | 81.7% | 0.425 | BTCUSDT | 212.10 bp | 100.0% | 24 | 190.7K | 69.6% |
| 68 | DRIFT | DRIFTUSDT (usdm-futures) | usdm-futures | 78.9% | 0.426 | BTCUSDT | 512.56 bp | 100.0% | 24 | 3.6M | 139.6% |
| 69 | BIO | BIOUSDT (spot) | spot, usdm-futures | 76.4% | 0.401 | BTCUSDT | 535.90 bp | 100.0% | 24 | 4.8M | 156.7% |
| 70 | C | CUSDT (spot) | spot, usdm-futures | 76.6% | 0.330 | BTCUSDT | 467.82 bp | 100.0% | 24 | 1.2M | 131.7% |
| 71 | PIVX | PIVXUSDT (spot) | spot | 76.6% | 0.308 | BTCUSDT | 445.64 bp | 100.0% | 48 | 337.8K | 131.5% |
| 72 | BICO | BICOUSDT (spot) | spot, usdm-futures | 77.9% | 0.438 | BTCUSDT | 412.67 bp | 100.0% | 24 | 416.7K | 136.1% |
| 73 | TST | TSTUSDT (spot) | spot, usdm-futures | 75.7% | 0.340 | BTCUSDT | 514.93 bp | 100.0% | 0 | 1.3M | 164.6% |
| 74 | BABY | BABYUSDT (spot) | spot, usdm-futures | 75.4% | 0.424 | BTCUSDT | 402.06 bp | 100.0% | 24 | 1.2M | 115.6% |
| 75 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 77.2% | 0.405 | BROCCOLIF3BUSDT | 395.32 bp | 100.0% | 24 | 844.6K | 236.3% |
| 76 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 75.1% | 0.442 | BTCUSDT | 437.45 bp | 100.0% | 24 | 1.3M | 160.0% |
| 77 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 75.6% | 0.440 | BTCUSDT | 388.54 bp | 100.0% | 24 | 525.7K | 121.3% |
| 78 | SPK | SPKUSDT (spot) | spot, usdm-futures | 76.0% | 0.440 | BTCUSDT | 384.20 bp | 100.0% | 0 | 2.4M | 109.3% |
| 79 | SKL | SKLUSDT (spot) | spot, usdm-futures | 76.7% | 0.463 | BTCUSDT | 379.02 bp | 100.0% | 24 | 788.7K | 117.5% |
| 80 | EDU | EDUUSDT (spot) | spot, usdm-futures | 73.7% | 0.408 | BTCUSDT | 395.52 bp | 100.0% | 24 | 1.1M | 124.7% |
| 81 | HYPE | HYPEUSDT (usdm-futures) | usdm-futures | 73.4% | 0.504 | BTCUSDT | 389.16 bp | 100.0% | 0 | 408.3M | 96.1% |
| 82 | PYR | PYRUSDT (spot) | spot | 73.6% | 0.401 | XNOUSDT | 368.46 bp | 100.0% | 48 | 1M | 123.7% |
| 83 | FLOW | FLOWUSDT (spot) | spot, usdm-futures | 74.6% | 0.480 | BTCUSDT | 364.93 bp | 100.0% | 24 | 972.5K | 112.0% |
| 84 | ORCA | ORCAUSDT (spot) | spot, usdm-futures | 73.0% | 0.452 | BTCUSDT | 368.91 bp | 100.0% | 24 | 812.5K | 108.9% |
| 85 | ONT | ONTUSDT (spot) | spot, usdm-futures | 72.9% | 0.440 | BTCUSDT | 350.19 bp | 100.0% | 24 | 623.5K | 109.0% |
| 86 | LSK | LSKUSDT (spot) | spot, usdm-futures | 74.6% | 0.468 | BTCUSDT | 327.05 bp | 100.0% | 48 | 347.2K | 102.5% |
| 87 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 72.6% | 0.464 | STGUSDT | 510.36 bp | 100.0% | 24 | 903.3K | 172.8% |
| 88 | G | GUSDT (spot) | spot, usdm-futures | 72.6% | 0.473 | BTCUSDT | 333.08 bp | 100.0% | 24 | 393.8K | 102.5% |
| 89 | PORTO | PORTOUSDT (spot) | spot | 74.2% | 0.492 | BTCUSDT | 278.27 bp | 100.0% | 24 | 291.9K | 87.3% |
| 90 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 72.0% | 0.341 | BANANAS31USDT | 633.23 bp | 100.0% | 24 | 2.6M | 182.3% |
| 91 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 71.4% | 0.485 | BTCUSDT | 408.44 bp | 100.0% | 24 | 1.2M | 122.6% |
| 92 | API3 | API3USDT (spot) | spot, usdm-futures | 71.3% | 0.493 | BTCUSDT | 340.05 bp | 100.0% | 0 | 960K | 98.0% |
| 93 | ASR | ASRUSDT (spot) | spot, usdm-futures | 71.6% | 0.451 | ATMUSDT | 305.76 bp | 100.0% | 24 | 755.1K | 103.2% |
| 94 | WIN | WINUSDT (spot) | spot | 72.8% | 0.429 | BTCUSDT | 239.25 bp | 100.0% | 24 | 307.3K | 79.5% |
| 95 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 69.3% | 0.410 | BTCUSDT | 584.99 bp | 100.0% | 0 | 4.3M | 165.7% |
| 96 | BROCCOLI714 | BROCCOLI714USDT (spot) | spot, usdm-futures | 69.5% | 0.466 | TSTUSDT | 484.90 bp | 100.0% | 24 | 1.6M | 157.4% |
| 97 | BERA | BERAUSDT (spot) | spot, usdm-futures | 70.0% | 0.484 | BTCUSDT | 449.86 bp | 100.0% | 48 | 2.9M | 135.9% |
| 98 | MOVR | MOVRUSDT (spot) | spot, usdm-futures | 68.9% | 0.442 | BTCUSDT | 400.15 bp | 100.0% | 24 | 784.4K | 132.4% |
| 99 | ZEC | ZECUSDT (spot) | spot, usdm-futures | 68.3% | 0.433 | BTCUSDT | 564.30 bp | 100.0% | 24 | 89M | 152.9% |
| 100 | LA | LAUSDT (spot) | spot, usdm-futures | 68.5% | 0.427 | BTCUSDT | 410.82 bp | 100.0% | 24 | 961.3K | 108.7% |
| 101 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 68.2% | 0.554 | BTCUSDT | 391.29 bp | 100.0% | 24 | 1.2M | 101.3% |
| 102 | RED | REDUSDT (spot) | spot, usdm-futures | 67.7% | 0.461 | BTCUSDT | 391.46 bp | 100.0% | 24 | 775.2K | 120.5% |
| 103 | CHEEMS | 1000CHEEMSUSDT (spot) | spot, usdm-futures | 68.1% | 0.505 | BTCUSDT | 390.98 bp | 100.0% | 24 | 1M | 106.1% |
| 104 | VIC | VICUSDT (spot) | spot, usdm-futures | 69.0% | 0.417 | FIDAUSDT | 380.69 bp | 100.0% | 24 | 437.8K | 121.2% |
| 105 | ENJ | ENJUSDT (spot) | spot, usdm-futures | 67.8% | 0.450 | BIOUSDT | 371.14 bp | 100.0% | 24 | 830.7K | 114.3% |
| 106 | OSMO | OSMOUSDT (spot) | spot | 67.7% | 0.445 | BTCUSDT | 357.28 bp | 100.0% | 24 | 312.7K | 122.1% |
| 107 | DGB | DGBUSDT (spot) | spot | 68.9% | 0.483 | BTCUSDT | 329.53 bp | 100.0% | 24 | 321.2K | 92.6% |
| 108 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 68.6% | 0.496 | BTCUSDT | 301.57 bp | 100.0% | 0 | 32.5M | 84.7% |
| 109 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 67.3% | 0.462 | BTCUSDT | 299.61 bp | 100.0% | 24 | 532.4K | 98.2% |
| 110 | ACX | ACXUSDT (spot) | spot, usdm-futures | 68.6% | 0.456 | BTCUSDT | 296.59 bp | 100.0% | 24 | 285.9K | 99.5% |
| 111 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 66.7% | 0.434 | VICUSDT | 451.85 bp | 100.0% | 24 | 833.5K | 136.2% |
| 112 | ACT | ACTUSDT (spot) | spot, usdm-futures | 66.4% | 0.446 | SWARMSUSDT | 441.49 bp | 100.0% | 48 | 1.9M | 124.9% |
| 113 | FORM | FORMUSDT (spot) | spot, usdm-futures | 65.8% | 0.440 | BTCUSDT | 480.95 bp | 100.0% | 24 | 2.3M | 138.3% |
| 114 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 65.8% | 0.461 | BTCUSDT | 406.30 bp | 100.0% | 0 | 2.3M | 103.4% |
| 115 | TWT | TWTUSDT (spot) | spot, usdm-futures | 66.6% | 0.446 | BTCUSDT | 278.85 bp | 100.0% | 24 | 1.1M | 81.4% |
| 116 | STRAX | STRAXUSDT (spot) | spot | 65.8% | 0.484 | PORTALUSDT | 278.76 bp | 100.0% | 24 | 374.1K | 93.5% |
| 117 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 66.4% | 0.444 | BTCUSDT | 271.24 bp | 100.0% | 24 | 1.3M | 81.5% |

## Diagnostics

- Basis size selected: 117
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.171
- Maximum pairwise absolute correlation: 0.554
- Mean whole-market projection R²: 82.0%
- Median whole-market projection R²: 80.2%
- 10th-percentile whole-market projection R²: 65.0%
- Minimum whole-market projection R²: 57.3%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 30.1% | 30.1% | 5.7% | 0.0% |
| 5 | 33.5% | 32.8% | 9.4% | 0.7% |
| 10 | 38.0% | 36.3% | 14.5% | 1.4% |
| 15 | 40.7% | 39.6% | 17.8% | 2.3% |
| 20 | 44.8% | 43.7% | 20.9% | 3.9% |
| 25 | 47.5% | 46.6% | 23.6% | 10.8% |
| 30 | 50.5% | 49.0% | 26.9% | 11.3% |
| 35 | 53.4% | 52.0% | 29.6% | 16.0% |
| 40 | 56.2% | 54.4% | 33.2% | 20.6% |
| 45 | 58.5% | 56.5% | 35.4% | 22.2% |
| 50 | 61.3% | 59.6% | 38.5% | 26.3% |
| 55 | 64.0% | 62.1% | 42.3% | 32.0% |
| 60 | 65.8% | 63.7% | 44.1% | 32.6% |
| 65 | 67.3% | 64.9% | 45.4% | 33.1% |
| 70 | 69.3% | 67.3% | 47.2% | 39.1% |
| 75 | 71.0% | 69.0% | 49.5% | 40.8% |
| 80 | 72.6% | 70.7% | 51.3% | 43.6% |
| 85 | 74.1% | 72.3% | 52.8% | 44.3% |
| 90 | 75.4% | 73.2% | 54.8% | 46.7% |
| 95 | 76.8% | 74.8% | 56.9% | 50.0% |
| 100 | 78.1% | 76.3% | 59.3% | 51.1% |
| 105 | 79.2% | 77.2% | 60.6% | 52.5% |
| 110 | 80.4% | 78.6% | 62.9% | 53.9% |
| 115 | 81.5% | 79.7% | 64.4% | 55.3% |
| 117 | 82.0% | 80.2% | 65.0% | 57.3% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | SIREN | H | PIPPIN | BULLA | FHE | SKYAI | AGT | VELVET | JELLYJELLY | M | BR | DEXE | B2 | BAN | STO | TNSR | ALCH | HOME | OG | ALPINE | JST | AIOT | B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.016 | 0.155 | 0.151 | 0.023 | 0.150 | 0.202 | 0.077 | 0.007 | 0.040 | 0.060 | 0.121 | 0.166 | 0.205 | 0.183 | 0.189 | 0.188 | 0.091 | 0.164 | 0.183 | 0.158 | 0.090 | 0.179 | 0.276 |
| SIREN | 0.016 | 1.000 | -0.020 | 0.070 | 0.013 | -0.013 | 0.046 | 0.102 | 0.069 | -0.001 | 0.128 | 0.031 | 0.069 | 0.020 | 0.039 | 0.018 | 0.033 | -0.022 | 0.037 | 0.002 | 0.064 | -0.005 | -0.105 | 0.016 |
| H | 0.155 | -0.020 | 1.000 | -0.011 | 0.033 | 0.050 | 0.044 | 0.055 | 0.026 | 0.143 | 0.065 | 0.093 | 0.002 | 0.020 | 0.063 | 0.032 | 0.061 | 0.030 | 0.070 | 0.024 | 0.106 | 0.025 | 0.001 | 0.046 |
| PIPPIN | 0.151 | 0.070 | -0.011 | 1.000 | 0.004 | 0.031 | 0.067 | 0.076 | 0.037 | 0.054 | 0.022 | 0.073 | 0.072 | 0.001 | 0.046 | 0.183 | 0.066 | 0.070 | -0.000 | 0.090 | 0.121 | 0.017 | 0.036 | 0.080 |
| BULLA | 0.023 | 0.013 | 0.033 | 0.004 | 1.000 | 0.053 | 0.040 | 0.097 | 0.025 | 0.017 | -0.016 | 0.021 | 0.007 | 0.018 | 0.056 | 0.031 | 0.067 | -0.022 | -0.047 | -0.003 | 0.026 | 0.033 | -0.027 | 0.030 |
| FHE | 0.150 | -0.013 | 0.050 | 0.031 | 0.053 | 1.000 | 0.153 | 0.058 | 0.063 | 0.005 | 0.008 | -0.036 | 0.046 | 0.033 | 0.038 | 0.056 | 0.028 | 0.054 | 0.045 | 0.042 | 0.092 | -0.026 | -0.002 | 0.033 |
| SKYAI | 0.202 | 0.046 | 0.044 | 0.067 | 0.040 | 0.153 | 1.000 | 0.089 | 0.039 | -0.054 | -0.027 | 0.100 | 0.117 | 0.138 | 0.026 | 0.044 | 0.099 | 0.071 | 0.072 | 0.052 | 0.111 | 0.053 | 0.190 | 0.097 |
| AGT | 0.077 | 0.102 | 0.055 | 0.076 | 0.097 | 0.058 | 0.089 | 1.000 | 0.095 | -0.032 | -0.050 | 0.017 | 0.004 | 0.015 | -0.021 | 0.088 | 0.133 | 0.027 | 0.038 | 0.052 | 0.098 | 0.039 | 0.033 | 0.073 |
| VELVET | 0.007 | 0.069 | 0.026 | 0.037 | 0.025 | 0.063 | 0.039 | 0.095 | 1.000 | -0.036 | -0.073 | 0.029 | 0.066 | 0.039 | -0.016 | 0.049 | 0.006 | -0.054 | -0.102 | -0.055 | 0.017 | 0.024 | 0.090 | -0.026 |
| JELLYJELLY | 0.040 | -0.001 | 0.143 | 0.054 | 0.017 | 0.005 | -0.054 | -0.032 | -0.036 | 1.000 | 0.049 | 0.037 | -0.025 | 0.018 | 0.058 | 0.037 | -0.036 | 0.029 | -0.035 | 0.001 | -0.003 | 0.008 | -0.028 | 0.080 |
| M | 0.060 | 0.128 | 0.065 | 0.022 | -0.016 | 0.008 | -0.027 | -0.050 | -0.073 | 0.049 | 1.000 | -0.008 | 0.023 | -0.009 | 0.017 | -0.004 | 0.008 | 0.028 | 0.064 | 0.034 | 0.070 | 0.040 | 0.019 | 0.069 |
| BR | 0.121 | 0.031 | 0.093 | 0.073 | 0.021 | -0.036 | 0.100 | 0.017 | 0.029 | 0.037 | -0.008 | 1.000 | 0.057 | 0.072 | 0.070 | -0.006 | 0.052 | 0.118 | -0.035 | 0.070 | -0.005 | -0.031 | 0.118 | 0.012 |
| DEXE | 0.166 | 0.069 | 0.002 | 0.072 | 0.007 | 0.046 | 0.117 | 0.004 | 0.066 | -0.025 | 0.023 | 0.057 | 1.000 | 0.046 | 0.023 | 0.038 | 0.061 | -0.003 | 0.089 | 0.006 | 0.032 | 0.049 | 0.013 | 0.070 |
| B2 | 0.205 | 0.020 | 0.020 | 0.001 | 0.018 | 0.033 | 0.138 | 0.015 | 0.039 | 0.018 | -0.009 | 0.072 | 0.046 | 1.000 | 0.082 | 0.027 | 0.150 | 0.030 | 0.068 | -0.014 | 0.086 | 0.028 | 0.090 | 0.109 |
| BAN | 0.183 | 0.039 | 0.063 | 0.046 | 0.056 | 0.038 | 0.026 | -0.021 | -0.016 | 0.058 | 0.017 | 0.070 | 0.023 | 0.082 | 1.000 | -0.006 | 0.021 | 0.004 | 0.065 | 0.074 | 0.048 | 0.072 | 0.060 | 0.120 |
| STO | 0.189 | 0.018 | 0.032 | 0.183 | 0.031 | 0.056 | 0.044 | 0.088 | 0.049 | 0.037 | -0.004 | -0.006 | 0.038 | 0.027 | -0.006 | 1.000 | 0.059 | -0.023 | 0.019 | 0.127 | 0.038 | -0.057 | -0.066 | 0.033 |
| TNSR | 0.188 | 0.033 | 0.061 | 0.066 | 0.067 | 0.028 | 0.099 | 0.133 | 0.006 | -0.036 | 0.008 | 0.052 | 0.061 | 0.150 | 0.021 | 0.059 | 1.000 | -0.000 | 0.079 | 0.041 | 0.083 | 0.013 | 0.041 | 0.054 |
| ALCH | 0.091 | -0.022 | 0.030 | 0.070 | -0.022 | 0.054 | 0.071 | 0.027 | -0.054 | 0.029 | 0.028 | 0.118 | -0.003 | 0.030 | 0.004 | -0.023 | -0.000 | 1.000 | -0.083 | 0.057 | 0.051 | -0.068 | 0.006 | 0.121 |
| HOME | 0.164 | 0.037 | 0.070 | -0.000 | -0.047 | 0.045 | 0.072 | 0.038 | -0.102 | -0.035 | 0.064 | -0.035 | 0.089 | 0.068 | 0.065 | 0.019 | 0.079 | -0.083 | 1.000 | -0.020 | 0.038 | 0.088 | -0.009 | 0.075 |
| OG | 0.183 | 0.002 | 0.024 | 0.090 | -0.003 | 0.042 | 0.052 | 0.052 | -0.055 | 0.001 | 0.034 | 0.070 | 0.006 | -0.014 | 0.074 | 0.127 | 0.041 | 0.057 | -0.020 | 1.000 | 0.146 | 0.009 | -0.030 | 0.020 |
| ALPINE | 0.158 | 0.064 | 0.106 | 0.121 | 0.026 | 0.092 | 0.111 | 0.098 | 0.017 | -0.003 | 0.070 | -0.005 | 0.032 | 0.086 | 0.048 | 0.038 | 0.083 | 0.051 | 0.038 | 0.146 | 1.000 | 0.001 | 0.022 | 0.082 |
| JST | 0.090 | -0.005 | 0.025 | 0.017 | 0.033 | -0.026 | 0.053 | 0.039 | 0.024 | 0.008 | 0.040 | -0.031 | 0.049 | 0.028 | 0.072 | -0.057 | 0.013 | -0.068 | 0.088 | 0.009 | 0.001 | 1.000 | -0.006 | 0.072 |
| AIOT | 0.179 | -0.105 | 0.001 | 0.036 | -0.027 | -0.002 | 0.190 | 0.033 | 0.090 | -0.028 | 0.019 | 0.118 | 0.013 | 0.090 | 0.060 | -0.066 | 0.041 | 0.006 | -0.009 | -0.030 | 0.022 | -0.006 | 1.000 | 0.063 |
| B | 0.276 | 0.016 | 0.046 | 0.080 | 0.030 | 0.033 | 0.097 | 0.073 | -0.026 | 0.080 | 0.069 | 0.012 | 0.070 | 0.109 | 0.120 | 0.033 | 0.054 | 0.121 | 0.075 | 0.020 | 0.082 | 0.072 | 0.063 | 1.000 |

The complete 117 × 117 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| TRX | TRXUSDT | 57.3% | 65.3% | SUNUSDT | 0.454 |
| QKC | QKCUSDT | 57.8% | 64.9% | BTCUSDT | 0.568 |
| TLM | TLMUSDT | 58.4% | 64.5% | BTCUSDT | 0.433 |
| KOMA | KOMAUSDT | 58.5% | 64.5% | BROCCOLI714USDT | 0.495 |
| CATI | CATIUSDT | 58.6% | 64.3% | BTCUSDT | 0.421 |
| TUT | TUTUSDT | 58.7% | 64.3% | TSTUSDT | 0.470 |
| PHA | PHAUSDT | 59.1% | 64.0% | BTCUSDT | 0.488 |
| QNT | QNTUSDT | 59.4% | 63.7% | BTCUSDT | 0.528 |
| FTT | FTTUSDT | 59.5% | 63.7% | LUNCUSDT | 0.480 |
| KERNEL | KERNELUSDT | 59.7% | 63.5% | API3USDT | 0.494 |
| YFI | YFIUSDT | 59.9% | 63.3% | BTCUSDT | 0.574 |
| KAIA | KAIAUSDT | 60.1% | 63.2% | BTCUSDT | 0.589 |
| JUV | JUVUSDT | 60.1% | 63.2% | PORTOUSDT | 0.505 |
| AVAAI | AVAAIUSDT | 60.4% | 62.9% | SWARMSUSDT | 0.523 |
| PYTH | PYTHUSDT | 60.7% | 62.7% | BTCUSDT | 0.569 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

