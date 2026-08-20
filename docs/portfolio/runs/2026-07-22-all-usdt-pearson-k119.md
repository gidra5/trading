# Binance portfolio basis

Generated 2026-07-23T17:47:54.993Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Daily log-return window: 2025-07-23 through 2026-07-22 (365 samples)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 128)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 412 eligible assets from 715 active continuously priced candidates
- Quality filter: at least 80.0% non-zero daily returns and a complete window
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 1.4B | 43.2% |
| 2 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 100.0% | 0.006 | BTCUSDT | 3.2M | 276.6% |
| 3 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.027 | VELVETUSDT | 2.7M | 384.7% |
| 4 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 99.8% | 0.041 | BTCUSDT | 7.5M | 258.8% |
| 5 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 99.7% | 0.068 | VELVETUSDT | 6.4M | 413.1% |
| 6 | JST | JSTUSDT (spot) | spot, usdm-futures | 99.5% | 0.091 | BTCUSDT | 2.2M | 61.1% |
| 7 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 99.1% | 0.091 | BTCUSDT | 3.7M | 151.1% |
| 8 | TA | TAUSDT (usdm-futures) | usdm-futures | 98.9% | 0.110 | BTCUSDT | 4.6M | 202.7% |
| 9 | M | MUSDT (usdm-futures) | usdm-futures | 98.7% | 0.126 | SIRENUSDT | 11.6M | 196.4% |
| 10 | BR | BRUSDT (usdm-futures) | usdm-futures | 98.2% | 0.121 | BTCUSDT | 1.5M | 191.2% |
| 11 | OG | OGUSDT (spot) | spot, usdm-futures | 97.9% | 0.184 | BTCUSDT | 2.1M | 131.0% |
| 12 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 97.7% | 0.151 | BTCUSDT | 4.8M | 253.3% |
| 13 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 97.6% | 0.189 | BTCUSDT | 1M | 204.7% |
| 14 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 97.4% | 0.151 | BTCUSDT | 25.1M | 293.2% |
| 15 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 97.4% | 0.167 | BTCUSDT | 1.3M | 225.6% |
| 16 | H | HUSDT (usdm-futures) | usdm-futures | 97.0% | 0.153 | BTCUSDT | 25.2M | 370.7% |
| 17 | BAN | BANUSDT (usdm-futures) | usdm-futures | 96.9% | 0.184 | BTCUSDT | 3.1M | 169.5% |
| 18 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 96.6% | 0.131 | TNSRUSDT | 3.5M | 222.3% |
| 19 | B2 | B2USDT (usdm-futures) | usdm-futures | 96.5% | 0.206 | BTCUSDT | 2.1M | 162.8% |
| 20 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.0% | 0.163 | BTCUSDT | 1.5M | 136.2% |
| 21 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 95.9% | 0.179 | BTCUSDT | 4.7M | 230.2% |
| 22 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 95.7% | 0.160 | BTCUSDT | 726.8K | 189.9% |
| 23 | TAC | TACUSDT (usdm-futures) | usdm-futures | 95.3% | 0.137 | BTCUSDT | 1.8M | 280.9% |
| 24 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 95.3% | 0.148 | HUSDT | 2M | 149.1% |
| 25 | STO | STOUSDT (spot) | spot, usdm-futures | 95.0% | 0.187 | BTCUSDT | 1.2M | 177.3% |
| 26 | B | BUSDT (usdm-futures) | usdm-futures | 94.1% | 0.271 | BTCUSDT | 4.8M | 213.9% |
| 27 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 93.3% | 0.189 | BTCUSDT | 2.3M | 153.6% |
| 28 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 92.9% | 0.222 | BTCUSDT | 3.6M | 152.6% |
| 29 | MERL | MERLUSDT (usdm-futures) | usdm-futures | 92.6% | 0.235 | BTCUSDT | 8.9M | 177.0% |
| 30 | PROM | PROMUSDT (spot) | spot, usdm-futures | 92.5% | 0.239 | DEXEUSDT | 1.2M | 121.2% |
| 31 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 92.1% | 0.234 | PIPPINUSDT | 4.7M | 215.2% |
| 32 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 91.9% | 0.272 | BTCUSDT | 7.4M | 187.1% |
| 33 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 91.7% | 0.271 | PIPPINUSDT | 9.3M | 205.6% |
| 34 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 91.0% | 0.204 | BTCUSDT | 18.6M | 317.7% |
| 35 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 90.8% | 0.293 | BTCUSDT | 1.1M | 147.0% |
| 36 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 90.6% | 0.297 | ALPINEUSDT | 2.9M | 160.2% |
| 37 | AIN | AINUSDT (usdm-futures) | usdm-futures | 90.5% | 0.187 | IDOLUSDT | 2.1M | 169.4% |
| 38 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 90.1% | 0.262 | BTCUSDT | 2.3M | 144.1% |
| 39 | DCR | DCRUSDT (spot) | spot | 89.8% | 0.333 | BTCUSDT | 436K | 120.0% |
| 40 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 89.5% | 0.221 | BTCUSDT | 1.8M | 161.0% |
| 41 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 89.0% | 0.225 | AINUSDT | 4.4M | 255.0% |
| 42 | ATM | ATMUSDT (spot) | spot | 88.9% | 0.229 | ALPINEUSDT | 702.4K | 104.5% |
| 43 | AWE | AWEUSDT (spot) | spot, usdm-futures | 88.6% | 0.328 | BTCUSDT | 480.2K | 94.7% |
| 44 | SUN | SUNUSDT (spot) | spot, usdm-futures | 88.4% | 0.267 | BTCUSDT | 1.1M | 63.4% |
| 45 | PAXG | PAXGUSDT (spot) | spot, usdm-futures | 87.0% | 0.332 | BTCUSDT | 23.6M | 28.9% |
| 46 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 86.4% | 0.361 | BTCUSDT | 2.1M | 122.0% |
| 47 | GUN | GUNUSDT (spot) | spot, usdm-futures | 85.8% | 0.322 | BTCUSDT | 1.6M | 146.1% |
| 48 | NIL | NILUSDT (spot) | spot, usdm-futures | 85.4% | 0.385 | BTCUSDT | 1.2M | 159.5% |
| 49 | STG | STGUSDT (spot) | spot, usdm-futures | 85.3% | 0.291 | BTCUSDT | 943K | 150.0% |
| 50 | XNO | XNOUSDT (spot) | spot | 85.1% | 0.321 | BTCUSDT | 217.6K | 104.7% |
| 51 | HEI | HEIUSDT (spot) | spot, usdm-futures | 84.5% | 0.233 | TAUSDT | 1M | 161.1% |
| 52 | PIVX | PIVXUSDT (spot) | spot | 84.2% | 0.309 | BTCUSDT | 348K | 131.1% |
| 53 | LUNC | LUNCUSDT (spot) | spot, usdm-futures | 83.5% | 0.376 | BTCUSDT | 2.6M | 118.7% |
| 54 | SYN | SYNUSDT (spot) | spot, usdm-futures | 83.2% | 0.262 | BULLAUSDT | 566.9K | 170.2% |
| 55 | SIGN | SIGNUSDT (spot) | spot, usdm-futures | 82.6% | 0.298 | BTCUSDT | 1M | 114.9% |
| 56 | DUSK | DUSKUSDT (spot) | spot, usdm-futures | 82.4% | 0.305 | BTCUSDT | 866.7K | 146.0% |
| 57 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 82.0% | 0.342 | BTCUSDT | 1.2M | 180.9% |
| 58 | NMR | NMRUSDT (spot) | spot, usdm-futures | 81.7% | 0.354 | TNSRUSDT | 887.5K | 123.3% |
| 59 | REQ | REQUSDT (spot) | spot | 81.5% | 0.426 | BTCUSDT | 193.3K | 69.3% |
| 60 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 81.2% | 0.404 | BROCCOLIF3BUSDT | 852.6K | 234.9% |
| 61 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 81.0% | 0.311 | BTCUSDT | 2.2M | 150.0% |
| 62 | VVV | VVVUSDT (usdm-futures) | usdm-futures | 80.9% | 0.440 | BTCUSDT | 12.4M | 147.7% |
| 63 | GPS | GPSUSDT (spot) | spot, usdm-futures | 80.4% | 0.326 | PARTIUSDT | 1.1M | 139.8% |
| 64 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 80.0% | 0.310 | TAUSDT | 3.6M | 167.5% |
| 65 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 79.5% | 0.312 | BTCUSDT | 2.6M | 161.8% |
| 66 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 79.5% | 0.336 | HUSDT | 1.5M | 162.4% |
| 67 | BIO | BIOUSDT (spot) | spot, usdm-futures | 79.0% | 0.402 | BTCUSDT | 4.7M | 156.3% |
| 68 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 78.8% | 0.313 | BTCUSDT | 1.7M | 150.5% |
| 69 | DRIFT | DRIFTUSDT (usdm-futures) | usdm-futures | 78.2% | 0.427 | BTCUSDT | 3.7M | 139.0% |
| 70 | BICO | BICOUSDT (spot) | spot, usdm-futures | 78.0% | 0.439 | BTCUSDT | 427.9K | 135.8% |
| 71 | RIF | RIFUSDT (spot) | spot, usdm-futures | 77.7% | 0.352 | DEXEUSDT | 531.7K | 139.0% |
| 72 | SKL | SKLUSDT (spot) | spot, usdm-futures | 77.2% | 0.463 | BTCUSDT | 808.3K | 117.1% |
| 73 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 77.1% | 0.427 | BTCUSDT | 946.6K | 101.3% |
| 74 | TST | TSTUSDT (spot) | spot, usdm-futures | 76.3% | 0.341 | BTCUSDT | 1.3M | 163.7% |
| 75 | C | CUSDT (spot) | spot, usdm-futures | 76.0% | 0.324 | BTCUSDT | 1.3M | 133.1% |
| 76 | BABY | BABYUSDT (spot) | spot, usdm-futures | 75.8% | 0.417 | BTCUSDT | 1.2M | 117.1% |
| 77 | LSK | LSKUSDT (spot) | spot, usdm-futures | 75.8% | 0.469 | BTCUSDT | 355K | 102.0% |
| 78 | SPK | SPKUSDT (spot) | spot, usdm-futures | 75.6% | 0.390 | SAHARAUSDT | 2.4M | 139.0% |
| 79 | FLOW | FLOWUSDT (spot) | spot, usdm-futures | 74.7% | 0.481 | BTCUSDT | 984.7K | 111.5% |
| 80 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 74.5% | 0.442 | BTCUSDT | 1.3M | 159.4% |
| 81 | STRAX | STRAXUSDT (spot) | spot | 74.4% | 0.437 | BTCUSDT | 384.1K | 93.5% |
| 82 | WIN | WINUSDT (spot) | spot | 74.3% | 0.429 | BTCUSDT | 310.3K | 79.1% |
| 83 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 73.9% | 0.440 | BTCUSDT | 535.8K | 120.9% |
| 84 | PORTO | PORTOUSDT (spot) | spot | 73.6% | 0.491 | BTCUSDT | 296.4K | 87.3% |
| 85 | HYPE | HYPEUSDT (usdm-futures) | usdm-futures | 73.4% | 0.502 | BTCUSDT | 404.6M | 95.9% |
| 86 | ORCA | ORCAUSDT (spot) | spot, usdm-futures | 73.3% | 0.452 | BTCUSDT | 838.4K | 108.5% |
| 87 | PYR | PYRUSDT (spot) | spot | 73.3% | 0.405 | XNOUSDT | 1M | 123.2% |
| 88 | RAD | RADUSDT (spot) | spot | 72.9% | 0.394 | BTCUSDT | 592.7K | 78.1% |
| 89 | ASR | ASRUSDT (spot) | spot, usdm-futures | 72.6% | 0.495 | ATMUSDT | 761.2K | 114.1% |
| 90 | ONT | ONTUSDT (spot) | spot, usdm-futures | 72.2% | 0.441 | BTCUSDT | 626.2K | 108.7% |
| 91 | EDU | EDUUSDT (spot) | spot, usdm-futures | 71.6% | 0.409 | BTCUSDT | 1.1M | 124.1% |
| 92 | G | GUSDT (spot) | spot, usdm-futures | 70.9% | 0.474 | BTCUSDT | 395K | 102.2% |
| 93 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 70.9% | 0.444 | BTCUSDT | 1.4M | 81.0% |
| 94 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 70.8% | 0.496 | BTCUSDT | 31.6M | 84.2% |
| 95 | FORM | FORMUSDT (spot) | spot, usdm-futures | 70.5% | 0.440 | BTCUSDT | 2.3M | 137.5% |
| 96 | TWT | TWTUSDT (spot) | spot, usdm-futures | 70.4% | 0.447 | BTCUSDT | 1.1M | 81.0% |
| 97 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 70.4% | 0.483 | BTCUSDT | 1.2M | 123.1% |
| 98 | KERNEL | KERNELUSDT (spot) | spot, usdm-futures | 70.0% | 0.464 | BTCUSDT | 1M | 122.2% |
| 99 | BERA | BERAUSDT (spot) | spot, usdm-futures | 69.8% | 0.484 | BTCUSDT | 3M | 135.2% |
| 100 | DGB | DGBUSDT (spot) | spot | 69.5% | 0.482 | BTCUSDT | 323.6K | 92.6% |
| 101 | VIC | VICUSDT (spot) | spot, usdm-futures | 69.2% | 0.423 | FIDAUSDT | 444K | 120.8% |
| 102 | CHEEMS | 1000CHEEMSUSDT (spot) | spot, usdm-futures | 68.8% | 0.506 | BTCUSDT | 999.8K | 105.8% |
| 103 | RED | REDUSDT (spot) | spot, usdm-futures | 68.4% | 0.459 | BTCUSDT | 784K | 120.4% |
| 104 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 68.3% | 0.461 | BTCUSDT | 554.5K | 98.0% |
| 105 | MOVR | MOVRUSDT (spot) | spot, usdm-futures | 68.2% | 0.440 | BTCUSDT | 792.1K | 132.6% |
| 106 | ACX | ACXUSDT (spot) | spot, usdm-futures | 68.1% | 0.456 | KERNELUSDT | 290.1K | 99.7% |
| 107 | API3 | API3USDT (spot) | spot, usdm-futures | 67.8% | 0.492 | BTCUSDT | 974.2K | 98.1% |
| 108 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 67.2% | 0.484 | STRAXUSDT | 915.8K | 172.4% |
| 109 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 66.9% | 0.410 | BTCUSDT | 4.3M | 165.3% |
| 110 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 66.8% | 0.437 | VICUSDT | 835.2K | 136.0% |
| 111 | TUT | TUTUSDT (spot) | spot, usdm-futures | 66.7% | 0.468 | TSTUSDT | 1.1M | 148.3% |
| 112 | ENJ | ENJUSDT (spot) | spot, usdm-futures | 66.6% | 0.455 | BIOUSDT | 842.4K | 114.2% |
| 113 | LA | LAUSDT (spot) | spot, usdm-futures | 66.2% | 0.426 | BTCUSDT | 972.9K | 108.0% |
| 114 | DOOD | DOODUSDT (usdm-futures) | usdm-futures | 65.9% | 0.490 | BTCUSDT | 2.2M | 129.9% |
| 115 | PHA | PHAUSDT (spot) | spot, usdm-futures | 65.6% | 0.488 | BTCUSDT | 1.1M | 121.8% |
| 116 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 65.0% | 0.464 | FIDAUSDT | 2.3M | 103.9% |
| 117 | ACT | ACTUSDT (spot) | spot, usdm-futures | 65.0% | 0.450 | SWARMSUSDT | 1.9M | 124.5% |
| 118 | OSMO | OSMOUSDT (spot) | spot | 64.9% | 0.466 | RADUSDT | 318.1K | 121.7% |
| 119 | QKC | QKCUSDT (spot) | spot | 64.8% | 0.568 | BTCUSDT | 207.9K | 66.6% |

## Diagnostics

- Basis size selected: 119
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.176
- Maximum pairwise absolute correlation: 0.568
- Mean whole-market projection R²: 82.1%
- Median whole-market projection R²: 80.1%
- 10th-percentile whole-market projection R²: 64.9%
- Minimum whole-market projection R²: 58.1%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 29.8% | 29.7% | 6.5% | 0.0% |
| 5 | 32.1% | 31.3% | 10.3% | 1.0% |
| 10 | 36.0% | 34.5% | 13.8% | 4.2% |
| 15 | 41.4% | 39.9% | 18.6% | 6.0% |
| 20 | 44.0% | 42.2% | 21.0% | 8.1% |
| 25 | 47.2% | 45.3% | 24.3% | 11.5% |
| 30 | 50.0% | 48.5% | 26.8% | 15.2% |
| 35 | 53.0% | 51.6% | 29.3% | 17.9% |
| 40 | 55.6% | 53.7% | 32.6% | 20.9% |
| 45 | 57.8% | 55.9% | 35.6% | 25.3% |
| 50 | 61.1% | 59.2% | 39.9% | 28.6% |
| 55 | 63.5% | 61.7% | 41.9% | 32.1% |
| 60 | 65.4% | 63.1% | 43.4% | 34.5% |
| 65 | 67.2% | 64.4% | 45.2% | 36.8% |
| 70 | 69.4% | 66.9% | 47.6% | 39.6% |
| 75 | 70.9% | 68.6% | 49.4% | 42.5% |
| 80 | 72.2% | 69.7% | 50.6% | 44.7% |
| 85 | 73.9% | 71.9% | 52.6% | 46.2% |
| 90 | 75.3% | 73.3% | 55.4% | 48.8% |
| 95 | 76.6% | 74.5% | 57.0% | 50.5% |
| 100 | 77.8% | 75.9% | 58.9% | 52.1% |
| 105 | 79.0% | 76.8% | 60.7% | 53.7% |
| 110 | 80.2% | 77.7% | 63.1% | 55.6% |
| 115 | 81.2% | 79.1% | 64.1% | 57.7% |
| 119 | 82.1% | 80.1% | 64.9% | 58.1% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | VELVET | BULLA | JELLYJELLY | SIREN | JST | ALCH | TA | M | BR | OG | FHE | TNSR | PIPPIN | DEXE | H | BAN | AGT | B2 | HOME | AIOT | ALPINE | TAC | IDOL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.006 | 0.023 | 0.041 | 0.017 | 0.091 | 0.091 | 0.110 | 0.057 | 0.121 | 0.184 | 0.151 | 0.189 | 0.151 | 0.167 | 0.153 | 0.184 | 0.075 | 0.206 | 0.163 | 0.179 | 0.160 | 0.137 | 0.140 |
| VELVET | 0.006 | 1.000 | 0.027 | -0.036 | 0.068 | 0.018 | -0.054 | 0.033 | -0.048 | 0.029 | -0.060 | 0.060 | 0.003 | 0.046 | 0.064 | 0.041 | -0.023 | 0.084 | 0.036 | -0.097 | 0.103 | 0.012 | 0.069 | 0.117 |
| BULLA | 0.023 | 0.027 | 1.000 | 0.017 | 0.013 | 0.032 | -0.022 | 0.042 | -0.014 | 0.021 | -0.002 | 0.053 | 0.067 | 0.006 | 0.007 | 0.035 | 0.056 | 0.095 | 0.018 | -0.046 | -0.024 | 0.026 | 0.035 | 0.041 |
| JELLYJELLY | 0.041 | -0.036 | 0.017 | 1.000 | -0.001 | 0.009 | 0.029 | -0.061 | 0.048 | 0.037 | 0.003 | 0.005 | -0.034 | 0.056 | -0.025 | 0.142 | 0.059 | -0.034 | 0.019 | -0.036 | -0.025 | -0.001 | 0.015 | 0.112 |
| SIREN | 0.017 | 0.068 | 0.013 | -0.001 | 1.000 | -0.005 | -0.022 | -0.021 | 0.126 | 0.031 | 0.002 | -0.012 | 0.033 | 0.070 | 0.069 | -0.020 | 0.039 | 0.100 | 0.020 | 0.037 | -0.104 | 0.064 | 0.042 | 0.083 |
| JST | 0.091 | 0.018 | 0.032 | 0.009 | -0.005 | 1.000 | -0.067 | 0.063 | 0.033 | -0.031 | 0.011 | -0.024 | 0.016 | 0.017 | 0.049 | 0.021 | 0.075 | 0.038 | 0.030 | 0.084 | -0.006 | 0.003 | 0.043 | 0.045 |
| ALCH | 0.091 | -0.054 | -0.022 | 0.029 | -0.022 | -0.067 | 1.000 | 0.018 | 0.026 | 0.118 | 0.058 | 0.054 | 0.001 | 0.071 | -0.003 | 0.029 | 0.005 | 0.026 | 0.030 | -0.083 | 0.007 | 0.052 | 0.072 | -0.005 |
| TA | 0.110 | 0.033 | 0.042 | -0.061 | -0.021 | 0.063 | 0.018 | 1.000 | 0.021 | 0.080 | 0.038 | 0.106 | 0.093 | 0.106 | 0.127 | 0.028 | -0.079 | 0.123 | 0.061 | 0.036 | 0.069 | 0.114 | 0.101 | 0.056 |
| M | 0.057 | -0.048 | -0.014 | 0.048 | 0.126 | 0.033 | 0.026 | 0.021 | 1.000 | -0.008 | 0.026 | 0.007 | 0.004 | 0.028 | 0.021 | 0.077 | 0.010 | -0.052 | -0.012 | 0.066 | 0.025 | 0.064 | -0.132 | -0.010 |
| BR | 0.121 | 0.029 | 0.021 | 0.037 | 0.031 | -0.031 | 0.118 | 0.080 | -0.008 | 1.000 | 0.070 | -0.036 | 0.052 | 0.073 | 0.057 | 0.092 | 0.070 | 0.017 | 0.072 | -0.035 | 0.117 | -0.005 | 0.060 | 0.007 |
| OG | 0.184 | -0.060 | -0.002 | 0.003 | 0.002 | 0.011 | 0.058 | 0.038 | 0.026 | 0.070 | 1.000 | 0.043 | 0.046 | 0.091 | 0.007 | 0.020 | 0.079 | 0.054 | -0.011 | -0.016 | -0.027 | 0.151 | 0.004 | 0.000 |
| FHE | 0.151 | 0.060 | 0.053 | 0.005 | -0.012 | -0.024 | 0.054 | 0.106 | 0.007 | -0.036 | 0.043 | 1.000 | 0.029 | 0.031 | 0.046 | 0.048 | 0.039 | 0.058 | 0.033 | 0.044 | -0.004 | 0.093 | 0.039 | -0.006 |
| TNSR | 0.189 | 0.003 | 0.067 | -0.034 | 0.033 | 0.016 | 0.001 | 0.093 | 0.004 | 0.052 | 0.046 | 0.029 | 1.000 | 0.069 | 0.061 | 0.059 | 0.024 | 0.131 | 0.152 | 0.078 | 0.044 | 0.085 | 0.069 | 0.029 |
| PIPPIN | 0.151 | 0.046 | 0.006 | 0.056 | 0.070 | 0.017 | 0.071 | 0.106 | 0.028 | 0.073 | 0.091 | 0.031 | 0.069 | 1.000 | 0.072 | -0.006 | 0.047 | 0.068 | 0.002 | -0.000 | 0.046 | 0.121 | 0.086 | -0.045 |
| DEXE | 0.167 | 0.064 | 0.007 | -0.025 | 0.069 | 0.049 | -0.003 | 0.127 | 0.021 | 0.057 | 0.007 | 0.046 | 0.061 | 0.072 | 1.000 | 0.001 | 0.024 | 0.004 | 0.046 | 0.089 | 0.013 | 0.033 | 0.027 | 0.058 |
| H | 0.153 | 0.041 | 0.035 | 0.142 | -0.020 | 0.021 | 0.029 | 0.028 | 0.077 | 0.092 | 0.020 | 0.048 | 0.059 | -0.006 | 0.001 | 1.000 | 0.058 | 0.049 | 0.018 | 0.072 | 0.010 | 0.102 | 0.118 | 0.148 |
| BAN | 0.184 | -0.023 | 0.056 | 0.059 | 0.039 | 0.075 | 0.005 | -0.079 | 0.010 | 0.070 | 0.079 | 0.039 | 0.024 | 0.047 | 0.024 | 0.058 | 1.000 | -0.019 | 0.083 | 0.064 | 0.058 | 0.051 | 0.022 | 0.052 |
| AGT | 0.075 | 0.084 | 0.095 | -0.034 | 0.100 | 0.038 | 0.026 | 0.123 | -0.052 | 0.017 | 0.054 | 0.058 | 0.131 | 0.068 | 0.004 | 0.049 | -0.019 | 1.000 | 0.014 | 0.041 | 0.021 | 0.098 | 0.130 | -0.010 |
| B2 | 0.206 | 0.036 | 0.018 | 0.019 | 0.020 | 0.030 | 0.030 | 0.061 | -0.012 | 0.072 | -0.011 | 0.033 | 0.152 | 0.002 | 0.046 | 0.018 | 0.083 | 0.014 | 1.000 | 0.067 | 0.089 | 0.088 | 0.032 | 0.031 |
| HOME | 0.163 | -0.097 | -0.046 | -0.036 | 0.037 | 0.084 | -0.083 | 0.036 | 0.066 | -0.035 | -0.016 | 0.044 | 0.078 | -0.000 | 0.089 | 0.072 | 0.064 | 0.041 | 0.067 | 1.000 | -0.006 | 0.040 | 0.067 | 0.103 |
| AIOT | 0.179 | 0.103 | -0.024 | -0.025 | -0.104 | -0.006 | 0.007 | 0.069 | 0.025 | 0.117 | -0.027 | -0.004 | 0.044 | 0.046 | 0.013 | 0.010 | 0.058 | 0.021 | 0.089 | -0.006 | 1.000 | 0.022 | 0.098 | 0.079 |
| ALPINE | 0.160 | 0.012 | 0.026 | -0.001 | 0.064 | 0.003 | 0.052 | 0.114 | 0.064 | -0.005 | 0.151 | 0.093 | 0.085 | 0.121 | 0.033 | 0.102 | 0.051 | 0.098 | 0.088 | 0.040 | 0.022 | 1.000 | 0.089 | 0.042 |
| TAC | 0.137 | 0.069 | 0.035 | 0.015 | 0.042 | 0.043 | 0.072 | 0.101 | -0.132 | 0.060 | 0.004 | 0.039 | 0.069 | 0.086 | 0.027 | 0.118 | 0.022 | 0.130 | 0.032 | 0.067 | 0.098 | 0.089 | 1.000 | 0.043 |
| IDOL | 0.140 | 0.117 | 0.041 | 0.112 | 0.083 | 0.045 | -0.005 | 0.056 | -0.010 | 0.007 | 0.000 | -0.006 | 0.029 | -0.045 | 0.058 | 0.148 | 0.052 | -0.010 | 0.031 | 0.103 | 0.079 | 0.042 | 0.043 | 1.000 |

The complete 119 × 119 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| ZEC | ZECUSDT | 58.1% | 64.7% | BTCUSDT | 0.433 |
| KOMA | KOMAUSDT | 58.6% | 64.4% | DOODUSDT | 0.426 |
| TRX | TRXUSDT | 58.8% | 64.2% | SUNUSDT | 0.458 |
| TLM | TLMUSDT | 59.0% | 64.0% | ARPAUSDT | 0.434 |
| QNT | QNTUSDT | 59.2% | 63.9% | BTCUSDT | 0.525 |
| DASH | DASHUSDT | 59.4% | 63.7% | DCRUSDT | 0.498 |
| KMNO | KMNOUSDT | 59.7% | 63.5% | BTCUSDT | 0.551 |
| CATI | CATIUSDT | 59.9% | 63.3% | BTCUSDT | 0.420 |
| BROCCOLI714 | BROCCOLI714USDT | 60.1% | 63.2% | DOODUSDT | 0.489 |
| YFI | YFIUSDT | 60.5% | 62.9% | BTCUSDT | 0.573 |
| PYTH | PYTHUSDT | 60.6% | 62.8% | BTCUSDT | 0.568 |
| ADX | ADXUSDT | 60.7% | 62.7% | BTCUSDT | 0.548 |
| AVAAI | AVAAIUSDT | 60.8% | 62.6% | SWARMSUSDT | 0.527 |
| KAIA | KAIAUSDT | 61.1% | 62.4% | BTCUSDT | 0.588 |
| FTT | FTTUSDT | 61.2% | 62.3% | LUNCUSDT | 0.484 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

