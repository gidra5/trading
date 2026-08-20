# Binance portfolio basis

Generated 2026-07-23T18:33:57.493Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2025-07-23 through 2026-07-22
- Sampling: 1d log returns (365 samples over 365 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 406 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.4B | 43.2% |
| 2 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 100.0% | 0.006 | BTCUSDT | 100.0% | 0 | 3.2M | 276.6% |
| 3 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.027 | VELVETUSDT | 100.0% | 24 | 2.7M | 384.7% |
| 4 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 99.8% | 0.041 | BTCUSDT | 100.0% | 0 | 7.5M | 258.8% |
| 5 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 99.7% | 0.068 | VELVETUSDT | 100.0% | 0 | 6.4M | 413.1% |
| 6 | JST | JSTUSDT (spot) | spot, usdm-futures | 99.5% | 0.091 | BTCUSDT | 100.0% | 24 | 2.2M | 61.1% |
| 7 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 99.1% | 0.091 | BTCUSDT | 100.0% | 24 | 3.7M | 151.1% |
| 8 | TA | TAUSDT (usdm-futures) | usdm-futures | 98.9% | 0.110 | BTCUSDT | 100.0% | 0 | 4.6M | 202.7% |
| 9 | M | MUSDT (usdm-futures) | usdm-futures | 98.7% | 0.126 | SIRENUSDT | 100.0% | 0 | 11.6M | 196.4% |
| 10 | BR | BRUSDT (usdm-futures) | usdm-futures | 98.2% | 0.121 | BTCUSDT | 100.0% | 48 | 1.5M | 191.2% |
| 11 | OG | OGUSDT (spot) | spot, usdm-futures | 97.9% | 0.184 | BTCUSDT | 100.0% | 24 | 2.1M | 131.0% |
| 12 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 97.7% | 0.151 | BTCUSDT | 100.0% | 0 | 4.8M | 253.3% |
| 13 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 97.6% | 0.189 | BTCUSDT | 100.0% | 24 | 1M | 204.7% |
| 14 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 97.4% | 0.151 | BTCUSDT | 100.0% | 24 | 25.1M | 293.2% |
| 15 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 97.4% | 0.167 | BTCUSDT | 100.0% | 24 | 1.3M | 225.6% |
| 16 | H | HUSDT (usdm-futures) | usdm-futures | 97.0% | 0.153 | BTCUSDT | 100.0% | 0 | 25.2M | 370.7% |
| 17 | BAN | BANUSDT (usdm-futures) | usdm-futures | 96.9% | 0.184 | BTCUSDT | 100.0% | 0 | 3.1M | 169.5% |
| 18 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 96.6% | 0.131 | TNSRUSDT | 100.0% | 0 | 3.5M | 222.3% |
| 19 | B2 | B2USDT (usdm-futures) | usdm-futures | 96.5% | 0.206 | BTCUSDT | 100.0% | 24 | 2.1M | 162.8% |
| 20 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.0% | 0.163 | BTCUSDT | 100.0% | 24 | 1.5M | 136.2% |
| 21 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 95.9% | 0.179 | BTCUSDT | 100.0% | 24 | 4.7M | 230.2% |
| 22 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 95.7% | 0.160 | BTCUSDT | 100.0% | 24 | 726.8K | 189.9% |
| 23 | TAC | TACUSDT (usdm-futures) | usdm-futures | 95.3% | 0.137 | BTCUSDT | 100.0% | 24 | 1.8M | 280.9% |
| 24 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 95.3% | 0.148 | HUSDT | 100.0% | 24 | 2M | 149.1% |
| 25 | STO | STOUSDT (spot) | spot, usdm-futures | 95.0% | 0.187 | BTCUSDT | 100.0% | 24 | 1.2M | 177.3% |
| 26 | B | BUSDT (usdm-futures) | usdm-futures | 94.1% | 0.271 | BTCUSDT | 100.0% | 0 | 4.8M | 213.9% |
| 27 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 93.3% | 0.189 | BTCUSDT | 100.0% | 48 | 2.3M | 153.6% |
| 28 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 92.9% | 0.222 | BTCUSDT | 100.0% | 24 | 3.6M | 152.6% |
| 29 | MERL | MERLUSDT (usdm-futures) | usdm-futures | 92.6% | 0.235 | BTCUSDT | 100.0% | 24 | 8.9M | 177.0% |
| 30 | PROM | PROMUSDT (spot) | spot, usdm-futures | 92.5% | 0.239 | DEXEUSDT | 100.0% | 24 | 1.2M | 121.2% |
| 31 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 92.1% | 0.234 | PIPPINUSDT | 100.0% | 24 | 4.7M | 215.2% |
| 32 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 91.9% | 0.272 | BTCUSDT | 100.0% | 0 | 7.4M | 187.1% |
| 33 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 91.7% | 0.271 | PIPPINUSDT | 100.0% | 0 | 9.3M | 205.6% |
| 34 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 91.0% | 0.204 | BTCUSDT | 100.0% | 24 | 18.6M | 317.7% |
| 35 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 90.8% | 0.293 | BTCUSDT | 100.0% | 0 | 1.1M | 147.0% |
| 36 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 90.6% | 0.297 | ALPINEUSDT | 100.0% | 24 | 2.9M | 160.2% |
| 37 | AIN | AINUSDT (usdm-futures) | usdm-futures | 90.5% | 0.187 | IDOLUSDT | 100.0% | 0 | 2.1M | 169.4% |
| 38 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 90.1% | 0.262 | BTCUSDT | 100.0% | 24 | 2.3M | 144.1% |
| 39 | DCR | DCRUSDT (spot) | spot | 89.8% | 0.333 | BTCUSDT | 100.0% | 24 | 436K | 120.0% |
| 40 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 89.5% | 0.221 | BTCUSDT | 100.0% | 24 | 1.8M | 161.0% |
| 41 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 89.0% | 0.225 | AINUSDT | 100.0% | 24 | 4.4M | 255.0% |
| 42 | ATM | ATMUSDT (spot) | spot | 88.9% | 0.229 | ALPINEUSDT | 100.0% | 24 | 702.4K | 104.5% |
| 43 | AWE | AWEUSDT (spot) | spot, usdm-futures | 88.6% | 0.328 | BTCUSDT | 100.0% | 24 | 480.2K | 94.7% |
| 44 | SUN | SUNUSDT (spot) | spot, usdm-futures | 88.4% | 0.267 | BTCUSDT | 100.0% | 24 | 1.1M | 63.4% |
| 45 | PAXG | PAXGUSDT (spot) | spot, usdm-futures | 87.0% | 0.332 | BTCUSDT | 100.0% | 0 | 23.6M | 28.9% |
| 46 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 86.4% | 0.361 | BTCUSDT | 100.0% | 0 | 2.1M | 122.0% |
| 47 | GUN | GUNUSDT (spot) | spot, usdm-futures | 85.8% | 0.322 | BTCUSDT | 100.0% | 24 | 1.6M | 146.1% |
| 48 | NIL | NILUSDT (spot) | spot, usdm-futures | 85.4% | 0.385 | BTCUSDT | 100.0% | 24 | 1.2M | 159.5% |
| 49 | STG | STGUSDT (spot) | spot, usdm-futures | 85.3% | 0.291 | BTCUSDT | 100.0% | 24 | 943K | 150.0% |
| 50 | XNO | XNOUSDT (spot) | spot | 85.1% | 0.321 | BTCUSDT | 100.0% | 48 | 217.6K | 104.7% |
| 51 | HEI | HEIUSDT (spot) | spot, usdm-futures | 84.5% | 0.233 | TAUSDT | 100.0% | 24 | 1M | 161.1% |
| 52 | PIVX | PIVXUSDT (spot) | spot | 84.2% | 0.309 | BTCUSDT | 100.0% | 48 | 348K | 131.1% |
| 53 | LUNC | LUNCUSDT (spot) | spot, usdm-futures | 83.5% | 0.376 | BTCUSDT | 100.0% | 0 | 2.6M | 118.7% |
| 54 | SYN | SYNUSDT (spot) | spot, usdm-futures | 83.2% | 0.262 | BULLAUSDT | 100.0% | 24 | 566.9K | 170.2% |
| 55 | SIGN | SIGNUSDT (spot) | spot, usdm-futures | 82.6% | 0.298 | BTCUSDT | 100.0% | 48 | 1M | 114.9% |
| 56 | DUSK | DUSKUSDT (spot) | spot, usdm-futures | 82.4% | 0.305 | BTCUSDT | 100.0% | 24 | 866.7K | 146.0% |
| 57 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 82.0% | 0.342 | BTCUSDT | 100.0% | 24 | 1.2M | 180.9% |
| 58 | NMR | NMRUSDT (spot) | spot, usdm-futures | 81.7% | 0.354 | TNSRUSDT | 100.0% | 24 | 887.5K | 123.3% |
| 59 | REQ | REQUSDT (spot) | spot | 81.5% | 0.426 | BTCUSDT | 100.0% | 24 | 193.3K | 69.3% |
| 60 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 81.2% | 0.404 | BROCCOLIF3BUSDT | 100.0% | 24 | 852.6K | 234.9% |
| 61 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 81.0% | 0.311 | BTCUSDT | 100.0% | 0 | 2.2M | 150.0% |
| 62 | VVV | VVVUSDT (usdm-futures) | usdm-futures | 80.9% | 0.440 | BTCUSDT | 100.0% | 24 | 12.4M | 147.7% |
| 63 | GPS | GPSUSDT (spot) | spot, usdm-futures | 80.4% | 0.326 | PARTIUSDT | 100.0% | 24 | 1.1M | 139.8% |
| 64 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 80.0% | 0.310 | TAUSDT | 100.0% | 24 | 3.6M | 167.5% |
| 65 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 79.5% | 0.312 | BTCUSDT | 100.0% | 24 | 2.6M | 161.8% |
| 66 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 79.5% | 0.336 | HUSDT | 100.0% | 24 | 1.5M | 162.4% |
| 67 | BIO | BIOUSDT (spot) | spot, usdm-futures | 79.0% | 0.402 | BTCUSDT | 100.0% | 24 | 4.7M | 156.3% |
| 68 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 78.8% | 0.313 | BTCUSDT | 100.0% | 0 | 1.7M | 150.5% |
| 69 | DRIFT | DRIFTUSDT (usdm-futures) | usdm-futures | 78.2% | 0.427 | BTCUSDT | 100.0% | 24 | 3.7M | 139.0% |
| 70 | BICO | BICOUSDT (spot) | spot, usdm-futures | 78.0% | 0.439 | BTCUSDT | 100.0% | 24 | 427.9K | 135.8% |
| 71 | RIF | RIFUSDT (spot) | spot, usdm-futures | 77.7% | 0.352 | DEXEUSDT | 100.0% | 24 | 531.7K | 139.0% |
| 72 | SKL | SKLUSDT (spot) | spot, usdm-futures | 77.2% | 0.463 | BTCUSDT | 100.0% | 24 | 808.3K | 117.1% |
| 73 | TST | TSTUSDT (spot) | spot, usdm-futures | 76.3% | 0.341 | BTCUSDT | 100.0% | 0 | 1.3M | 163.7% |
| 74 | C | CUSDT (spot) | spot, usdm-futures | 76.0% | 0.324 | BTCUSDT | 100.0% | 24 | 1.3M | 133.1% |
| 75 | LSK | LSKUSDT (spot) | spot, usdm-futures | 75.9% | 0.469 | BTCUSDT | 100.0% | 48 | 355K | 102.0% |
| 76 | BABY | BABYUSDT (spot) | spot, usdm-futures | 75.8% | 0.417 | BTCUSDT | 100.0% | 24 | 1.2M | 117.1% |
| 77 | SPK | SPKUSDT (spot) | spot, usdm-futures | 75.6% | 0.390 | SAHARAUSDT | 100.0% | 0 | 2.4M | 139.0% |
| 78 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 75.3% | 0.440 | BTCUSDT | 100.0% | 24 | 535.8K | 120.9% |
| 79 | FLOW | FLOWUSDT (spot) | spot, usdm-futures | 74.8% | 0.481 | BTCUSDT | 100.0% | 24 | 984.7K | 111.5% |
| 80 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 74.5% | 0.442 | BTCUSDT | 100.0% | 24 | 1.3M | 159.4% |
| 81 | STRAX | STRAXUSDT (spot) | spot | 74.4% | 0.437 | BTCUSDT | 100.0% | 24 | 384.1K | 93.5% |
| 82 | WIN | WINUSDT (spot) | spot | 74.3% | 0.429 | BTCUSDT | 100.0% | 24 | 310.3K | 79.1% |
| 83 | EDU | EDUUSDT (spot) | spot, usdm-futures | 73.7% | 0.409 | BTCUSDT | 100.0% | 24 | 1.1M | 124.1% |
| 84 | PORTO | PORTOUSDT (spot) | spot | 73.6% | 0.491 | BTCUSDT | 100.0% | 24 | 296.4K | 87.3% |
| 85 | HYPE | HYPEUSDT (usdm-futures) | usdm-futures | 73.4% | 0.502 | BTCUSDT | 100.0% | 0 | 404.6M | 95.9% |
| 86 | PYR | PYRUSDT (spot) | spot | 73.3% | 0.405 | XNOUSDT | 100.0% | 48 | 1M | 123.2% |
| 87 | ORCA | ORCAUSDT (spot) | spot, usdm-futures | 73.3% | 0.452 | BTCUSDT | 100.0% | 24 | 838.4K | 108.5% |
| 88 | ONT | ONTUSDT (spot) | spot, usdm-futures | 72.6% | 0.441 | BTCUSDT | 100.0% | 24 | 626.2K | 108.7% |
| 89 | G | GUSDT (spot) | spot, usdm-futures | 72.0% | 0.474 | BTCUSDT | 100.0% | 24 | 395K | 102.2% |
| 90 | ASR | ASRUSDT (spot) | spot, usdm-futures | 72.0% | 0.495 | ATMUSDT | 100.0% | 24 | 761.2K | 114.1% |
| 91 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 71.4% | 0.496 | BTCUSDT | 100.0% | 0 | 31.6M | 84.2% |
| 92 | API3 | API3USDT (spot) | spot, usdm-futures | 71.3% | 0.492 | BTCUSDT | 100.0% | 0 | 974.2K | 98.1% |
| 93 | OSMO | OSMOUSDT (spot) | spot | 71.2% | 0.446 | BTCUSDT | 100.0% | 24 | 318.1K | 121.7% |
| 94 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 70.9% | 0.483 | BTCUSDT | 100.0% | 24 | 1.2M | 123.1% |
| 95 | TWT | TWTUSDT (spot) | spot, usdm-futures | 70.5% | 0.447 | BTCUSDT | 100.0% | 24 | 1.1M | 81.0% |
| 96 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 70.4% | 0.444 | BTCUSDT | 100.0% | 24 | 1.4M | 81.0% |
| 97 | CHEEMS | 1000CHEEMSUSDT (spot) | spot, usdm-futures | 70.1% | 0.506 | BTCUSDT | 100.0% | 24 | 999.8K | 105.8% |
| 98 | FORM | FORMUSDT (spot) | spot, usdm-futures | 69.7% | 0.440 | BTCUSDT | 100.0% | 24 | 2.3M | 137.5% |
| 99 | DGB | DGBUSDT (spot) | spot | 69.6% | 0.482 | BTCUSDT | 100.0% | 24 | 323.6K | 92.6% |
| 100 | BERA | BERAUSDT (spot) | spot, usdm-futures | 69.2% | 0.484 | BTCUSDT | 100.0% | 48 | 3M | 135.2% |
| 101 | KERNEL | KERNELUSDT (spot) | spot, usdm-futures | 68.7% | 0.464 | BTCUSDT | 100.0% | 24 | 1M | 122.2% |
| 102 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 68.7% | 0.461 | BTCUSDT | 100.0% | 24 | 554.5K | 98.0% |
| 103 | ACX | ACXUSDT (spot) | spot, usdm-futures | 68.3% | 0.456 | KERNELUSDT | 100.0% | 24 | 290.1K | 99.7% |
| 104 | RED | REDUSDT (spot) | spot, usdm-futures | 68.0% | 0.459 | BTCUSDT | 100.0% | 24 | 784K | 120.4% |
| 105 | VIC | VICUSDT (spot) | spot, usdm-futures | 67.8% | 0.423 | FIDAUSDT | 100.0% | 24 | 444K | 120.8% |
| 106 | ENJ | ENJUSDT (spot) | spot, usdm-futures | 67.7% | 0.455 | BIOUSDT | 100.0% | 24 | 842.4K | 114.2% |
| 107 | LA | LAUSDT (spot) | spot, usdm-futures | 67.2% | 0.426 | BTCUSDT | 100.0% | 24 | 972.9K | 108.0% |
| 108 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 67.0% | 0.410 | BTCUSDT | 100.0% | 0 | 4.3M | 165.3% |
| 109 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 66.9% | 0.484 | STRAXUSDT | 100.0% | 24 | 915.8K | 172.4% |
| 110 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 66.7% | 0.437 | VICUSDT | 100.0% | 24 | 835.2K | 136.0% |
| 111 | MOVR | MOVRUSDT (spot) | spot, usdm-futures | 66.6% | 0.440 | BTCUSDT | 100.0% | 24 | 792.1K | 132.6% |
| 112 | TUT | TUTUSDT (spot) | spot, usdm-futures | 66.4% | 0.468 | TSTUSDT | 100.0% | 24 | 1.1M | 148.3% |
| 113 | DOOD | DOODUSDT (usdm-futures) | usdm-futures | 65.9% | 0.490 | BTCUSDT | 100.0% | 24 | 2.2M | 129.9% |
| 114 | PHA | PHAUSDT (spot) | spot, usdm-futures | 65.6% | 0.488 | BTCUSDT | 100.0% | 24 | 1.1M | 121.8% |
| 115 | TLM | TLMUSDT (spot) | spot, usdm-futures | 65.2% | 0.434 | ARPAUSDT | 100.0% | 24 | 672.6K | 130.5% |
| 116 | ZEC | ZECUSDT (spot) | spot, usdm-futures | 64.9% | 0.433 | BTCUSDT | 100.0% | 24 | 87.5M | 152.0% |
| 117 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 64.9% | 0.464 | FIDAUSDT | 100.0% | 0 | 2.3M | 103.9% |

## Diagnostics

- Basis size selected: 117
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.172
- Maximum pairwise absolute correlation: 0.506
- Mean whole-market projection R²: 81.9%
- Median whole-market projection R²: 80.1%
- 10th-percentile whole-market projection R²: 65.1%
- Minimum whole-market projection R²: 58.0%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 29.9% | 29.7% | 6.1% | 0.0% |
| 5 | 32.2% | 31.6% | 9.9% | 1.0% |
| 10 | 36.1% | 34.5% | 13.8% | 4.2% |
| 15 | 41.5% | 40.2% | 18.3% | 6.0% |
| 20 | 44.1% | 42.6% | 21.0% | 8.1% |
| 25 | 47.4% | 46.1% | 24.1% | 11.5% |
| 30 | 50.1% | 48.6% | 26.9% | 15.2% |
| 35 | 53.2% | 52.0% | 29.3% | 17.9% |
| 40 | 55.8% | 53.8% | 32.6% | 20.9% |
| 45 | 58.0% | 56.2% | 35.6% | 25.3% |
| 50 | 61.3% | 59.8% | 39.9% | 28.6% |
| 55 | 63.8% | 61.9% | 42.1% | 32.1% |
| 60 | 65.7% | 63.1% | 44.0% | 34.5% |
| 65 | 67.4% | 64.5% | 45.6% | 36.8% |
| 70 | 69.6% | 67.6% | 47.8% | 39.6% |
| 75 | 71.2% | 68.9% | 49.8% | 42.5% |
| 80 | 72.6% | 70.4% | 51.2% | 44.7% |
| 85 | 74.1% | 72.0% | 53.0% | 46.3% |
| 90 | 75.5% | 73.3% | 55.3% | 49.0% |
| 95 | 76.9% | 75.3% | 57.4% | 50.4% |
| 100 | 78.1% | 76.3% | 59.4% | 52.8% |
| 105 | 79.3% | 77.0% | 61.4% | 54.2% |
| 110 | 80.4% | 78.3% | 63.1% | 55.7% |
| 115 | 81.4% | 79.7% | 64.5% | 57.8% |
| 117 | 81.9% | 80.1% | 65.1% | 58.0% |

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

The complete 117 × 117 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| QKC | QKCUSDT | 58.0% | 64.8% | BTCUSDT | 0.568 |
| KOMA | KOMAUSDT | 58.2% | 64.7% | DOODUSDT | 0.426 |
| ACT | ACTUSDT | 58.3% | 64.6% | SWARMSUSDT | 0.450 |
| KMNO | KMNOUSDT | 59.1% | 64.0% | BTCUSDT | 0.551 |
| QNT | QNTUSDT | 59.1% | 63.9% | BTCUSDT | 0.525 |
| BROCCOLI714 | BROCCOLI714USDT | 59.2% | 63.9% | DOODUSDT | 0.489 |
| TRX | TRXUSDT | 59.3% | 63.8% | SUNUSDT | 0.458 |
| CATI | CATIUSDT | 59.6% | 63.6% | BTCUSDT | 0.420 |
| ADX | ADXUSDT | 60.3% | 63.0% | BTCUSDT | 0.548 |
| YFI | YFIUSDT | 60.3% | 63.0% | BTCUSDT | 0.573 |
| AVAAI | AVAAIUSDT | 60.4% | 62.9% | SWARMSUSDT | 0.527 |
| PYTH | PYTHUSDT | 60.4% | 62.9% | BTCUSDT | 0.568 |
| FTT | FTTUSDT | 60.5% | 62.9% | LUNCUSDT | 0.484 |
| RPL | RPLUSDT | 60.8% | 62.6% | BTCUSDT | 0.504 |
| KAIA | KAIAUSDT | 60.9% | 62.5% | BTCUSDT | 0.588 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

