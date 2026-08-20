# Binance portfolio basis

Generated 2026-07-23T19:08:51.577Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-05-24T00:00Z through 2026-07-22T20:00Z
- Sampling: exactly 360 4h log returns (60 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 1.0% of the maximum unexplained variance, select the largest mean absolute 4h return
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 581 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 61.80 bp | 100.0% | 0 | 1.2B | 41.3% |
| 2 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 99.8% | 0.066 | BTCUSDT | 787.95 bp | 100.0% | 4 | 53.3M | 772.5% |
| 3 | LAB | LABUSDT (usdm-futures) | usdm-futures | 100.0% | 0.016 | BTCUSDT | 648.53 bp | 100.0% | 0 | 357.6M | 492.8% |
| 4 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 99.9% | 0.032 | LABUSDT | 571.67 bp | 100.0% | 0 | 45.5M | 482.9% |
| 5 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 99.5% | 0.083 | BTCUSDT | 469.39 bp | 100.0% | 4 | 10.6M | 363.4% |
| 6 | HEI | HEIUSDT (spot) | spot, usdm-futures | 99.8% | 0.048 | EVAAUSDT | 409.08 bp | 100.0% | 4 | 3.9M | 329.6% |
| 7 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 99.9% | 0.032 | BTCUSDT | 401.62 bp | 100.0% | 4 | 10.2M | 313.4% |
| 8 | RIF | RIFUSDT (spot) | spot, usdm-futures | 99.4% | 0.089 | BTCUSDT | 385.12 bp | 100.0% | 4 | 2M | 273.7% |
| 9 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 99.6% | 0.069 | BTCUSDT | 370.24 bp | 100.0% | 4 | 3.8M | 280.9% |
| 10 | M | MUSDT (usdm-futures) | usdm-futures | 99.8% | 0.045 | BTCUSDT | 306.18 bp | 100.0% | 4 | 4.5M | 400.9% |
| 11 | BAS | BASUSDT (usdm-futures) | usdm-futures | 99.2% | 0.090 | BTCUSDT | 351.26 bp | 100.0% | 0 | 7.8M | 268.4% |
| 12 | US | USUSDT (usdm-futures) | usdm-futures | 99.0% | 0.075 | VELVETUSDT | 368.84 bp | 100.0% | 4 | 8.6M | 272.4% |
| 13 | B | BUSDT (usdm-futures) | usdm-futures | 99.5% | 0.048 | LABUSDT | 288.56 bp | 100.0% | 4 | 4.9M | 281.3% |
| 14 | IN | INUSDT (usdm-futures) | usdm-futures | 99.1% | 0.084 | BASUSDT | 279.77 bp | 100.0% | 0 | 6.9M | 366.0% |
| 15 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 98.4% | 0.091 | BTCUSDT | 523.71 bp | 100.0% | 4 | 18.2M | 441.6% |
| 16 | UB | UBUSDT (usdm-futures) | usdm-futures | 98.3% | 0.105 | BTCUSDT | 403.73 bp | 100.0% | 4 | 27.5M | 271.9% |
| 17 | H | HUSDT (usdm-futures) | usdm-futures | 98.2% | 0.113 | BTCUSDT | 579.76 bp | 100.0% | 0 | 46.8M | 574.4% |
| 18 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 98.2% | 0.118 | RIFUSDT | 302.43 bp | 100.0% | 0 | 4.8M | 339.0% |
| 19 | BEL | BELUSDT (spot) | spot, usdm-futures | 98.0% | 0.158 | BTCUSDT | 241.93 bp | 100.0% | 4 | 723.1K | 198.7% |
| 20 | GPS | GPSUSDT (spot) | spot, usdm-futures | 98.3% | 0.083 | EVAAUSDT | 153.70 bp | 100.0% | 4 | 691.7K | 127.5% |
| 21 | BABY | BABYUSDT (spot) | spot, usdm-futures | 98.0% | 0.109 | BTCUSDT | 164.57 bp | 100.0% | 4 | 637K | 184.6% |
| 22 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 98.3% | 0.082 | ESPORTSUSDT | 140.41 bp | 100.0% | 4 | 3.7M | 98.4% |
| 23 | DODO | DODOUSDT (spot) | spot | 97.5% | 0.105 | INUSDT | 213.61 bp | 100.0% | 4 | 623.1K | 174.2% |
| 24 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 97.5% | 0.093 | ALLOUSDT | 157.74 bp | 100.0% | 4 | 2.6M | 108.7% |
| 25 | Q | QUSDT (usdm-futures) | usdm-futures | 97.4% | 0.145 | EVAAUSDT | 155.95 bp | 100.0% | 4 | 1.9M | 110.5% |
| 26 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 97.8% | 0.109 | INUSDT | 55.08 bp | 100.0% | 4 | 20.6M | 41.1% |
| 27 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 97.3% | 0.099 | ESPORTSUSDT | 400.00 bp | 100.0% | 0 | 27.3M | 279.0% |
| 28 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 97.1% | 0.107 | BTCUSDT | 316.45 bp | 100.0% | 4 | 2.1M | 253.0% |
| 29 | STRAX | STRAXUSDT (spot) | spot | 96.9% | 0.156 | BTCUSDT | 191.29 bp | 100.0% | 4 | 680K | 194.2% |
| 30 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 97.0% | 0.113 | RIFUSDT | 171.95 bp | 100.0% | 4 | 687.1K | 182.6% |
| 31 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 97.1% | 0.125 | ESPORTSUSDT | 135.63 bp | 100.0% | 4 | 1.5M | 91.3% |
| 32 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 96.3% | 0.108 | BTCUSDT | 227.10 bp | 100.0% | 4 | 2.3M | 174.7% |
| 33 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 96.7% | 0.140 | BTCUSDT | 134.41 bp | 100.0% | 8 | 1.1M | 123.4% |
| 34 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 95.9% | 0.101 | EVAAUSDT | 375.27 bp | 100.0% | 0 | 11.4M | 291.3% |
| 35 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 96.1% | 0.132 | USUSDT | 313.58 bp | 100.0% | 4 | 1.8M | 350.6% |
| 36 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 95.8% | 0.157 | VELVETUSDT | 242.17 bp | 100.0% | 8 | 1.2M | 400.6% |
| 37 | HD | HDUSDT (usdm-futures) | usdm-futures | 95.7% | 0.112 | LABUSDT | 40.80 bp | 100.0% | 4 | 395.1K | 32.0% |
| 38 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 95.4% | 0.138 | ESPORTSUSDT | 328.13 bp | 100.0% | 0 | 4.9M | 233.9% |
| 39 | KGST | KGSTUSDT (spot) | spot | 95.8% | 0.110 | BABYUSDT | 5.59 bp | 100.0% | 24 | 75.5K | 4.5% |
| 40 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 95.0% | 0.114 | GUAUSDT | 369.33 bp | 100.0% | 8 | 3.1M | 276.4% |
| 41 | STG | STGUSDT (spot) | spot, usdm-futures | 94.5% | 0.124 | JCTUSDT | 304.22 bp | 100.0% | 4 | 1.5M | 244.6% |
| 42 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 94.8% | 0.143 | VELVETUSDT | 226.45 bp | 100.0% | 0 | 3.6M | 162.5% |
| 43 | ATM | ATMUSDT (spot) | spot | 94.2% | 0.129 | BASUSDT | 226.41 bp | 100.0% | 4 | 971.8K | 202.5% |
| 44 | DGB | DGBUSDT (spot) | spot | 94.6% | 0.177 | BTCUSDT | 154.86 bp | 100.0% | 8 | 105.3K | 120.7% |
| 45 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 93.7% | 0.147 | BTCUSDT | 360.74 bp | 100.0% | 0 | 9.9M | 286.5% |
| 46 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 93.5% | 0.165 | BTCUSDT | 190.80 bp | 100.0% | 4 | 1.8M | 146.0% |
| 47 | U | UUSDT (spot) | spot | 94.0% | 0.149 | KGSTUSDT | 0.91 bp | 100.0% | 24 | 15.9M | 0.6% |
| 48 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 93.3% | 0.149 | VELVETUSDT | 504.44 bp | 100.0% | 0 | 71.1M | 339.3% |
| 49 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 92.7% | 0.118 | AKEUSDT | 333.22 bp | 100.0% | 0 | 3.6M | 253.5% |
| 50 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 92.8% | 0.168 | ALLOUSDT | 252.52 bp | 100.0% | 0 | 4.2M | 195.5% |
| 51 | PIVX | PIVXUSDT (spot) | spot | 93.1% | 0.207 | VELVETUSDT | 216.91 bp | 100.0% | 8 | 205K | 245.8% |
| 52 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 92.8% | 0.186 | BTCUSDT | 167.94 bp | 100.0% | 0 | 1.2M | 148.1% |
| 53 | CATI | CATIUSDT (spot) | spot, usdm-futures | 92.8% | 0.165 | BTCUSDT | 155.82 bp | 100.0% | 12 | 408.3K | 117.3% |
| 54 | SYN | SYNUSDT (spot) | spot, usdm-futures | 92.2% | 0.116 | STGUSDT | 458.82 bp | 100.0% | 4 | 6.2M | 326.5% |
| 55 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 91.7% | 0.178 | GPSUSDT | 329.57 bp | 100.0% | 0 | 3.8M | 249.0% |
| 56 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 91.7% | 0.143 | BTCUSDT | 302.56 bp | 100.0% | 4 | 13.6M | 216.2% |
| 57 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 91.1% | 0.131 | BSBUSDT | 443.90 bp | 100.0% | 0 | 32.9M | 319.1% |
| 58 | DRIFT | DRIFTUSDT (usdm-futures) | usdm-futures | 91.1% | 0.263 | BTCUSDT | 182.37 bp | 100.0% | 4 | 1.9M | 155.7% |
| 59 | TA | TAUSDT (usdm-futures) | usdm-futures | 91.2% | 0.158 | QUSDT | 175.80 bp | 100.0% | 4 | 2.3M | 133.8% |
| 60 | ACE | ACEUSDT (spot) | spot, usdm-futures | 91.3% | 0.237 | BTCUSDT | 163.81 bp | 100.0% | 8 | 359K | 154.7% |
| 61 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 90.6% | 0.287 | BTCUSDT | 178.21 bp | 100.0% | 4 | 1.3M | 139.9% |
| 62 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 90.4% | 0.129 | 龙虾USDT | 318.64 bp | 100.0% | 4 | 12.4M | 230.2% |
| 63 | XEC | XECUSDT (spot) | spot, usdm-futures | 90.7% | 0.267 | BTCUSDT | 146.21 bp | 100.0% | 8 | 187.6K | 124.1% |
| 64 | B2 | B2USDT (usdm-futures) | usdm-futures | 90.2% | 0.129 | DGBUSDT | 146.86 bp | 100.0% | 4 | 1.4M | 101.1% |
| 65 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 90.4% | 0.128 | UUSDT | 48.23 bp | 100.0% | 4 | 544.6K | 40.9% |
| 66 | TLM | TLMUSDT (spot) | spot, usdm-futures | 89.5% | 0.203 | BTCUSDT | 323.65 bp | 100.0% | 8 | 793.1K | 294.4% |
| 67 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 89.7% | 0.210 | BTCUSDT | 269.38 bp | 100.0% | 4 | 13.1M | 209.5% |
| 68 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 89.6% | 0.207 | STRAXUSDT | 257.53 bp | 100.0% | 4 | 4.5M | 174.9% |
| 69 | ERA | ERAUSDT (spot) | spot, usdm-futures | 89.6% | 0.284 | BTCUSDT | 154.07 bp | 100.0% | 4 | 340.3K | 168.0% |
| 70 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 88.6% | 0.207 | NATGASUSDT | 302.46 bp | 100.0% | 4 | 1.5M | 317.6% |
| 71 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 88.7% | 0.174 | BTCUSDT | 286.62 bp | 100.0% | 4 | 4.1M | 196.8% |
| 72 | ID | IDUSDT (spot) | spot, usdm-futures | 88.7% | 0.166 | STGUSDT | 244.93 bp | 100.0% | 12 | 1.5M | 174.5% |
| 73 | BICO | BICOUSDT (spot) | spot, usdm-futures | 88.8% | 0.251 | BTCUSDT | 226.15 bp | 100.0% | 16 | 629.4K | 205.7% |
| 74 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 88.3% | 0.188 | BTCUSDT | 220.89 bp | 100.0% | 4 | 772.5K | 184.9% |
| 75 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 88.4% | 0.201 | HUSDT | 181.88 bp | 100.0% | 4 | 1.9M | 236.6% |
| 76 | PROM | PROMUSDT (spot) | spot, usdm-futures | 88.4% | 0.230 | BTCUSDT | 149.68 bp | 100.0% | 8 | 276K | 144.1% |
| 77 | QUICK | QUICKUSDT (spot) | spot | 87.8% | 0.211 | PIVXUSDT | 177.02 bp | 100.0% | 4 | 204.1K | 155.7% |
| 78 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 87.6% | 0.231 | BTCUSDT | 166.50 bp | 100.0% | 8 | 592K | 154.9% |
| 79 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 87.9% | 0.200 | BTCUSDT | 157.00 bp | 100.0% | 4 | 3.7M | 103.7% |
| 80 | PORTO | PORTOUSDT (spot) | spot | 87.5% | 0.236 | AKEUSDT | 140.71 bp | 100.0% | 4 | 302.6K | 128.1% |
| 81 | ACT | ACTUSDT (spot) | spot, usdm-futures | 86.9% | 0.273 | BTCUSDT | 194.45 bp | 100.0% | 4 | 504K | 184.0% |
| 82 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 86.5% | 0.153 | TRIAUSDT | 220.99 bp | 100.0% | 4 | 1.7M | 156.5% |
| 83 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 86.7% | 0.279 | BTCUSDT | 157.43 bp | 100.0% | 8 | 764K | 151.8% |
| 84 | REQ | REQUSDT (spot) | spot | 86.3% | 0.309 | BTCUSDT | 125.44 bp | 100.0% | 12 | 118.1K | 114.3% |
| 85 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 86.6% | 0.207 | BTCUSDT | 117.46 bp | 100.0% | 4 | 423.2K | 113.5% |
| 86 | JST | JSTUSDT (spot) | spot, usdm-futures | 86.7% | 0.165 | BTCUSDT | 90.19 bp | 100.0% | 4 | 3.5M | 63.6% |
| 87 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 85.7% | 0.206 | POWERUSDT | 244.31 bp | 100.0% | 4 | 4.3M | 172.6% |
| 88 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 85.9% | 0.204 | BTCUSDT | 227.90 bp | 100.0% | 0 | 1.9M | 154.9% |
| 89 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 85.7% | 0.352 | BTCUSDT | 167.48 bp | 100.0% | 4 | 1.9M | 130.0% |
| 90 | APR | APRUSDT (usdm-futures) | usdm-futures | 85.2% | 0.143 | MAGMAUSDT | 199.38 bp | 100.0% | 4 | 3.3M | 138.5% |
| 91 | AT | ATUSDT (spot) | spot, usdm-futures | 85.2% | 0.166 | JCTUSDT | 145.09 bp | 100.0% | 4 | 356.4K | 111.6% |
| 92 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 84.6% | 0.183 | QUSDT | 197.74 bp | 100.0% | 4 | 2.3M | 143.6% |
| 93 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 84.6% | 0.306 | BTCUSDT | 132.04 bp | 100.0% | 4 | 663.8K | 81.5% |
| 94 | QKC | QKCUSDT (spot) | spot | 84.5% | 0.341 | BTCUSDT | 109.09 bp | 100.0% | 4 | 93.6K | 96.2% |
| 95 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 84.0% | 0.235 | BTCUSDT | 227.84 bp | 100.0% | 0 | 269.4K | 207.7% |
| 96 | MITO | MITOUSDT (spot) | spot, usdm-futures | 83.3% | 0.182 | 龙虾USDT | 197.71 bp | 100.0% | 4 | 920.7K | 145.6% |
| 97 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 83.3% | 0.330 | BTCUSDT | 194.41 bp | 100.0% | 4 | 2.6M | 126.9% |
| 98 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 83.6% | 0.298 | BTCUSDT | 123.93 bp | 100.0% | 4 | 178.2K | 138.6% |
| 99 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 83.1% | 0.212 | MANTAUSDT | 170.03 bp | 100.0% | 0 | 5.2M | 145.3% |
| 100 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 82.7% | 0.207 | BTCUSDT | 183.65 bp | 100.0% | 4 | 2.1M | 135.1% |
| 101 | SXT | SXTUSDT (spot) | spot, usdm-futures | 82.6% | 0.310 | BTCUSDT | 186.99 bp | 100.0% | 4 | 618.4K | 140.4% |
| 102 | AWE | AWEUSDT (spot) | spot, usdm-futures | 82.6% | 0.197 | TNSRUSDT | 125.14 bp | 100.0% | 4 | 381.2K | 82.7% |
| 103 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 82.4% | 0.178 | RIFUSDT | 158.76 bp | 100.0% | 4 | 3.3M | 121.4% |
| 104 | RPL | RPLUSDT (spot) | spot, usdm-futures | 81.9% | 0.342 | BTCUSDT | 158.51 bp | 100.0% | 16 | 125.3K | 136.0% |
| 105 | SKL | SKLUSDT (spot) | spot, usdm-futures | 82.0% | 0.374 | BTCUSDT | 155.85 bp | 100.0% | 8 | 410.9K | 126.9% |
| 106 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 81.9% | 0.249 | HDUSDT | 43.74 bp | 100.0% | 0 | 6.4M | 34.6% |
| 107 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 82.0% | 0.228 | BTCUSDT | 39.49 bp | 100.0% | 4 | 414.6K | 31.4% |
| 108 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 80.9% | 0.240 | BTCUSDT | 312.78 bp | 100.0% | 4 | 4.3M | 229.8% |
| 109 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 80.2% | 0.158 | HUMAUSDT | 489.14 bp | 100.0% | 0 | 22.6M | 350.8% |
| 110 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 80.3% | 0.293 | KAITOUSDT | 301.54 bp | 100.0% | 4 | 2.9M | 324.4% |
| 111 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 79.9% | 0.171 | ESPORTSUSDT | 258.87 bp | 100.0% | 0 | 4.1M | 171.6% |
| 112 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 79.9% | 0.192 | TAUSDT | 216.27 bp | 100.0% | 8 | 1.6M | 152.4% |
| 113 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 79.6% | 0.242 | BTCUSDT | 209.71 bp | 100.0% | 0 | 1.1M | 141.3% |
| 114 | STAR | STARUSDT (usdm-futures) | usdm-futures | 79.1% | 0.202 | BTCUSDT | 246.74 bp | 100.0% | 4 | 1.8M | 164.7% |
| 115 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 79.4% | 0.195 | BTCUSDT | 194.28 bp | 100.0% | 0 | 1.5M | 130.4% |
| 116 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 78.8% | 0.295 | PROMUSDT | 160.86 bp | 100.0% | 4 | 2.9M | 100.4% |
| 117 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 78.7% | 0.246 | ESPORTSUSDT | 358.00 bp | 100.0% | 4 | 16M | 335.7% |
| 118 | KITE | KITEUSDT (spot) | spot, usdm-futures | 78.9% | 0.301 | BTCUSDT | 157.74 bp | 100.0% | 4 | 2.5M | 100.7% |
| 119 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 78.4% | 0.226 | DRIFTUSDT | 189.29 bp | 100.0% | 4 | 3.2M | 135.3% |
| 120 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 78.2% | 0.253 | BTCUSDT | 171.61 bp | 100.0% | 4 | 2.9M | 159.9% |
| 121 | G | GUSDT (spot) | spot, usdm-futures | 78.2% | 0.344 | BTCUSDT | 155.94 bp | 100.0% | 12 | 326.1K | 116.1% |
| 122 | CITY | CITYUSDT (spot) | spot | 78.2% | 0.258 | BTCUSDT | 107.83 bp | 100.0% | 8 | 440K | 95.3% |
| 123 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 77.7% | 0.235 | ZBTUSDT | 273.61 bp | 100.0% | 4 | 2.6M | 198.6% |
| 124 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 77.3% | 0.187 | HMSTRUSDT | 246.90 bp | 100.0% | 4 | 1.8M | 188.1% |
| 125 | XNO | XNOUSDT (spot) | spot | 77.4% | 0.280 | DGBUSDT | 161.08 bp | 100.0% | 8 | 61.1K | 138.7% |
| 126 | XAN | XANUSDT (usdm-futures) | usdm-futures | 76.8% | 0.199 | CHIPUSDT | 235.07 bp | 100.0% | 4 | 2.8M | 157.9% |
| 127 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 76.3% | 0.321 | BTCUSDT | 256.09 bp | 100.0% | 4 | 2.4M | 165.9% |
| 128 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 76.3% | 0.326 | BTCUSDT | 128.31 bp | 100.0% | 0 | 35.1M | 88.1% |
| 129 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 76.5% | 0.217 | BTCUSDT | 124.10 bp | 100.0% | 4 | 827.1K | 90.6% |
| 130 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 76.4% | 0.186 | JSTUSDT | 26.40 bp | 100.0% | 4 | 444.2K | 19.1% |
| 131 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 75.7% | 0.277 | BTCUSDT | 234.07 bp | 100.0% | 8 | 922.1K | 172.6% |
| 132 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 75.5% | 0.292 | AIOTUSDT | 230.57 bp | 100.0% | 0 | 3.4M | 217.3% |
| 133 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 75.2% | 0.203 | XNYUSDT | 169.02 bp | 100.0% | 4 | 1.4M | 114.4% |
| 134 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 74.8% | 0.195 | KITEUSDT | 229.28 bp | 100.0% | 0 | 2.7M | 160.7% |
| 135 | USTC | USTCUSDT (spot) | spot, usdm-futures | 75.0% | 0.385 | BTCUSDT | 111.52 bp | 100.0% | 4 | 367K | 78.6% |
| 136 | OPG | OPGUSDT (spot) | spot, usdm-futures | 74.2% | 0.211 | ICNTUSDT | 253.21 bp | 100.0% | 4 | 4.3M | 191.2% |
| 137 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 74.3% | 0.192 | BABYUSDT | 177.05 bp | 100.0% | 4 | 1.2M | 131.8% |
| 138 | DIS | DISUSDT (usdm-futures) | usdm-futures | 74.3% | 0.319 | HDUSDT | 36.80 bp | 100.0% | 4 | 272K | 27.3% |
| 139 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 73.9% | 0.365 | BTCUSDT | 146.87 bp | 100.0% | 4 | 424.1K | 102.4% |
| 140 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 73.2% | 0.288 | BTCUSDT | 181.62 bp | 100.0% | 4 | 2.5M | 152.0% |
| 141 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 72.8% | 0.282 | SXTUSDT | 180.36 bp | 100.0% | 4 | 3.8M | 138.7% |
| 142 | OSMO | OSMOUSDT (spot) | spot | 73.0% | 0.272 | CHIPUSDT | 179.16 bp | 100.0% | 8 | 466.3K | 147.0% |
| 143 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 72.9% | 0.340 | BTCUSDT | 178.17 bp | 100.0% | 4 | 1.9M | 135.3% |
| 144 | ONE | ONEUSDT (spot) | spot, usdm-futures | 72.7% | 0.425 | BTCUSDT | 145.43 bp | 100.0% | 12 | 192.3K | 118.1% |
| 145 | T | TUSDT (spot) | spot, usdm-futures | 72.9% | 0.356 | BTCUSDT | 139.06 bp | 100.0% | 8 | 234.3K | 115.3% |
| 146 | TAC | TACUSDT (usdm-futures) | usdm-futures | 71.8% | 0.418 | SPELLUSDT | 381.48 bp | 100.0% | 4 | 4.1M | 616.4% |
| 147 | JTO | JTOUSDT (spot) | spot, usdm-futures | 71.9% | 0.330 | BTCUSDT | 239.05 bp | 100.0% | 0 | 5.8M | 159.3% |
| 148 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 71.3% | 0.319 | BASUSDT | 232.61 bp | 100.0% | 4 | 2.3M | 172.1% |
| 149 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 71.0% | 0.328 | COAIUSDT | 187.21 bp | 100.0% | 4 | 3.5M | 164.8% |
| 150 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 71.0% | 0.200 | FOGOUSDT | 155.30 bp | 100.0% | 4 | 1.1M | 102.7% |
| 151 | BAN | BANUSDT (usdm-futures) | usdm-futures | 70.7% | 0.225 | SPACEUSDT | 170.64 bp | 100.0% | 0 | 2.6M | 116.7% |
| 152 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 70.9% | 0.276 | PORTOUSDT | 148.64 bp | 100.0% | 4 | 2.4M | 112.1% |
| 153 | SENT | SENTUSDT (spot) | spot, usdm-futures | 70.2% | 0.262 | ALCHUSDT | 179.60 bp | 100.0% | 4 | 1.1M | 120.8% |
| 154 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 70.3% | 0.164 | 龙虾USDT | 146.14 bp | 100.0% | 0 | 2.5M | 107.1% |
| 155 | SKY | SKYUSDT (spot) | spot, usdm-futures | 70.3% | 0.437 | BTCUSDT | 102.03 bp | 100.0% | 4 | 1.2M | 66.4% |
| 156 | BR | BRUSDT (usdm-futures) | usdm-futures | 69.4% | 0.244 | SENTUSDT | 232.19 bp | 100.0% | 0 | 1.8M | 173.3% |
| 157 | PYR | PYRUSDT (spot) | spot | 69.6% | 0.321 | XPINUSDT | 175.86 bp | 100.0% | 16 | 662.6K | 141.7% |
| 158 | FLNC | FLNCUSDT (usdm-futures) | usdm-futures | 69.5% | 0.279 | CSCOUSDT | 172.90 bp | 100.0% | 4 | 5.2M | 135.6% |
| 159 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 68.8% | 0.373 | FIDAUSDT | 218.99 bp | 100.0% | 4 | 1.9M | 138.6% |
| 160 | BZ | BZUSDT (usdm-futures) | usdm-futures | 68.9% | 0.338 | HDUSDT | 70.85 bp | 100.0% | 4 | 172.3M | 46.2% |
| 161 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 68.7% | 0.366 | AMZNUSDT | 38.53 bp | 100.0% | 4 | 9.3M | 31.5% |
| 162 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 68.0% | 0.317 | TNSRUSDT | 218.28 bp | 100.0% | 4 | 2.5M | 157.5% |
| 163 | ZORA | ZORAUSDT (usdm-futures) | usdm-futures | 67.7% | 0.350 | BTCUSDT | 149.47 bp | 100.0% | 4 | 2.4M | 103.4% |
| 164 | DCR | DCRUSDT (spot) | spot | 68.0% | 0.371 | XECUSDT | 110.30 bp | 100.0% | 4 | 200.1K | 86.0% |
| 165 | XLM | XLMUSDT (spot) | spot, usdm-futures, coinm-futures | 67.4% | 0.374 | BTCUSDT | 167.62 bp | 100.0% | 12 | 19.3M | 123.6% |
| 166 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 67.5% | 0.268 | BTCUSDT | 86.89 bp | 100.0% | 8 | 559K | 71.0% |
| 167 | THE | THEUSDT (spot) | spot, usdm-futures | 66.7% | 0.361 | RPLUSDT | 171.59 bp | 100.0% | 8 | 546.5K | 146.3% |
| 168 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 66.7% | 0.375 | BTCUSDT | 147.43 bp | 100.0% | 4 | 1.9M | 93.3% |
| 169 | ARK | ARKUSDT (spot) | spot, usdm-futures | 66.6% | 0.455 | BTCUSDT | 116.05 bp | 100.0% | 8 | 72.9K | 103.9% |
| 170 | V | VUSDT (usdm-futures) | usdm-futures | 66.7% | 0.271 | DISUSDT | 39.84 bp | 100.0% | 4 | 301.7K | 29.1% |
| 171 | PSG | PSGUSDT (spot) | spot | 66.1% | 0.348 | ATMUSDT | 105.92 bp | 100.0% | 12 | 686.2K | 82.2% |
| 172 | SUN | SUNUSDT (spot) | spot, usdm-futures | 66.2% | 0.312 | BTCUSDT | 46.52 bp | 100.0% | 4 | 735K | 31.6% |
| 173 | WLD | WLDUSDT (spot) | spot, usdm-futures | 65.0% | 0.358 | BTCUSDT | 242.47 bp | 100.0% | 0 | 43.1M | 176.7% |
| 174 | ON | ONUSDT (usdm-futures) | usdm-futures | 65.1% | 0.343 | PROMUSDT | 233.78 bp | 100.0% | 4 | 1.5M | 168.7% |
| 175 | AI | AIUSDT (spot) | spot | 64.9% | 0.246 | BTCUSDT | 145.38 bp | 100.0% | 16 | 409.8K | 111.7% |
| 176 | IQ | IQUSDT (spot) | spot | 65.0% | 0.420 | BTCUSDT | 83.95 bp | 100.0% | 8 | 69.2K | 60.4% |
| 177 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 64.2% | 0.314 | BTCUSDT | 163.31 bp | 100.0% | 4 | 1.4M | 105.8% |
| 178 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 64.4% | 0.536 | BTCUSDT | 127.81 bp | 100.0% | 8 | 969.9K | 96.2% |
| 179 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 63.8% | 0.223 | FOLKSUSDT | 269.61 bp | 100.0% | 0 | 2.5M | 211.9% |
| 180 | BANK | BANKUSDT (spot) | spot, usdm-futures | 63.2% | 0.372 | DEXEUSDT | 319.04 bp | 100.0% | 8 | 511.5K | 338.0% |
| 181 | LIT | LITUSDT (usdm-futures) | usdm-futures | 63.0% | 0.407 | BTCUSDT | 256.00 bp | 100.0% | 0 | 50.9M | 162.8% |
| 182 | AIN | AINUSDT (usdm-futures) | usdm-futures | 63.1% | 0.227 | SWARMSUSDT | 250.32 bp | 100.0% | 0 | 1.3M | 181.5% |

## Diagnostics

- Basis size selected: 182
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.065
- Maximum pairwise absolute correlation: 0.536
- Mean whole-market projection R²: 82.6%
- Median whole-market projection R²: 80.4%
- 10th-percentile whole-market projection R²: 65.0%
- Minimum whole-market projection R²: 60.5%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.1% | 20.3% | 1.1% | 0.0% |
| 5 | 24.7% | 21.3% | 2.7% | 0.2% |
| 10 | 26.7% | 22.6% | 4.5% | 0.6% |
| 15 | 29.0% | 25.5% | 6.2% | 2.4% |
| 20 | 31.1% | 27.7% | 8.0% | 3.3% |
| 25 | 33.9% | 30.2% | 10.0% | 4.3% |
| 30 | 35.9% | 32.7% | 11.8% | 5.7% |
| 35 | 37.8% | 34.1% | 13.4% | 7.7% |
| 40 | 40.0% | 36.1% | 15.6% | 10.0% |
| 45 | 42.2% | 38.3% | 17.2% | 11.7% |
| 50 | 44.1% | 39.8% | 19.0% | 13.3% |
| 55 | 46.0% | 42.0% | 20.8% | 15.8% |
| 60 | 48.3% | 43.8% | 22.4% | 17.2% |
| 65 | 50.2% | 45.9% | 24.1% | 19.3% |
| 70 | 52.0% | 48.0% | 25.6% | 21.1% |
| 75 | 53.8% | 49.8% | 27.8% | 21.8% |
| 80 | 55.7% | 51.9% | 29.9% | 24.3% |
| 85 | 57.7% | 54.1% | 31.6% | 24.9% |
| 90 | 59.2% | 55.1% | 33.7% | 27.1% |
| 95 | 60.8% | 57.2% | 35.4% | 30.0% |
| 100 | 62.5% | 59.1% | 37.5% | 31.2% |
| 105 | 64.1% | 60.4% | 39.2% | 32.8% |
| 110 | 65.5% | 62.2% | 41.2% | 35.6% |
| 115 | 66.8% | 63.3% | 42.8% | 37.3% |
| 120 | 68.1% | 64.2% | 44.4% | 38.7% |
| 125 | 69.6% | 65.6% | 46.6% | 41.0% |
| 130 | 70.8% | 66.7% | 48.9% | 42.7% |
| 135 | 72.1% | 68.3% | 50.2% | 44.5% |
| 140 | 73.4% | 69.9% | 52.1% | 46.4% |
| 145 | 74.7% | 71.5% | 53.7% | 48.2% |
| 150 | 75.9% | 72.7% | 55.5% | 49.7% |
| 155 | 76.9% | 73.6% | 57.0% | 51.5% |
| 160 | 78.0% | 74.9% | 58.2% | 52.8% |
| 165 | 79.0% | 76.1% | 59.9% | 54.5% |
| 170 | 80.1% | 76.9% | 61.5% | 56.2% |
| 175 | 81.2% | 78.7% | 62.9% | 57.7% |
| 180 | 82.2% | 79.8% | 64.2% | 59.9% |
| 182 | 82.6% | 80.4% | 65.0% | 60.5% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | ESPORTS | LAB | VELVET | EVAA | HEI | ALLO | RIF | HOME | M | BAS | US | B | IN | GUA | UB | H | DEXE | BEL | GPS | BABY | STABLE | DODO | XPIN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.066 | 0.016 | 0.004 | 0.083 | 0.037 | 0.032 | 0.089 | 0.069 | 0.045 | 0.090 | 0.036 | 0.027 | 0.042 | 0.091 | 0.105 | 0.113 | 0.029 | 0.158 | 0.063 | 0.109 | 0.058 | 0.069 | 0.045 |
| ESPORTS | 0.066 | 1.000 | -0.005 | -0.029 | -0.029 | 0.019 | 0.030 | -0.028 | 0.005 | -0.023 | 0.014 | 0.058 | 0.028 | -0.028 | 0.037 | -0.013 | -0.027 | -0.079 | -0.021 | -0.003 | 0.015 | 0.082 | -0.032 | 0.009 |
| LAB | 0.016 | -0.005 | 1.000 | 0.032 | -0.022 | 0.023 | -0.021 | -0.010 | -0.008 | 0.033 | 0.005 | -0.057 | -0.048 | 0.031 | 0.051 | 0.034 | 0.048 | -0.026 | 0.017 | 0.013 | 0.013 | -0.018 | -0.021 | -0.055 |
| VELVET | 0.004 | -0.029 | 0.032 | 1.000 | -0.022 | 0.015 | -0.002 | 0.015 | -0.028 | 0.009 | 0.084 | -0.075 | -0.048 | 0.030 | 0.060 | 0.046 | 0.092 | 0.016 | -0.008 | -0.061 | -0.061 | -0.052 | 0.028 | -0.008 |
| EVAA | 0.083 | -0.029 | -0.022 | -0.022 | 1.000 | 0.048 | 0.007 | -0.007 | 0.026 | -0.001 | 0.005 | 0.013 | 0.029 | -0.004 | -0.048 | 0.006 | 0.047 | 0.064 | 0.086 | 0.083 | 0.035 | -0.016 | -0.055 | 0.041 |
| HEI | 0.037 | 0.019 | 0.023 | 0.015 | 0.048 | 1.000 | 0.017 | 0.037 | 0.032 | 0.022 | 0.004 | -0.018 | -0.025 | 0.034 | 0.022 | 0.060 | 0.032 | 0.011 | 0.054 | -0.004 | 0.010 | -0.023 | -0.020 | 0.017 |
| ALLO | 0.032 | 0.030 | -0.021 | -0.002 | 0.007 | 0.017 | 1.000 | 0.009 | 0.018 | 0.016 | 0.001 | 0.034 | -0.030 | 0.010 | -0.014 | 0.008 | -0.003 | 0.046 | -0.001 | -0.038 | 0.033 | 0.018 | -0.025 | -0.093 |
| RIF | 0.089 | -0.028 | -0.010 | 0.015 | -0.007 | 0.037 | 0.009 | 1.000 | 0.020 | -0.014 | 0.037 | -0.008 | -0.009 | 0.053 | 0.074 | 0.024 | 0.015 | 0.118 | 0.026 | 0.028 | 0.041 | 0.026 | -0.007 | 0.050 |
| HOME | 0.069 | 0.005 | -0.008 | -0.028 | 0.026 | 0.032 | 0.018 | 0.020 | 1.000 | -0.001 | 0.003 | -0.035 | -0.003 | -0.001 | -0.002 | 0.026 | 0.080 | 0.035 | 0.033 | 0.011 | -0.043 | 0.047 | 0.040 | 0.035 |
| M | 0.045 | -0.023 | 0.033 | 0.009 | -0.001 | 0.022 | 0.016 | -0.014 | -0.001 | 1.000 | -0.006 | 0.039 | 0.038 | -0.013 | -0.045 | 0.036 | 0.033 | -0.013 | 0.004 | 0.026 | -0.030 | -0.079 | 0.009 | -0.043 |
| BAS | 0.090 | 0.014 | 0.005 | 0.084 | 0.005 | 0.004 | 0.001 | 0.037 | 0.003 | -0.006 | 1.000 | -0.053 | -0.028 | -0.084 | 0.004 | 0.058 | 0.022 | -0.010 | 0.026 | 0.061 | 0.060 | -0.010 | -0.065 | 0.007 |
| US | 0.036 | 0.058 | -0.057 | -0.075 | 0.013 | -0.018 | 0.034 | -0.008 | -0.035 | 0.039 | -0.053 | 1.000 | 0.015 | 0.007 | -0.073 | 0.047 | -0.048 | -0.020 | 0.004 | -0.033 | -0.087 | -0.002 | 0.015 | -0.052 |
| B | 0.027 | 0.028 | -0.048 | -0.048 | 0.029 | -0.025 | -0.030 | -0.009 | -0.003 | 0.038 | -0.028 | 0.015 | 1.000 | 0.019 | -0.035 | 0.044 | 0.032 | 0.045 | 0.007 | 0.033 | 0.016 | 0.062 | 0.026 | -0.025 |
| IN | 0.042 | -0.028 | 0.031 | 0.030 | -0.004 | 0.034 | 0.010 | 0.053 | -0.001 | -0.013 | -0.084 | 0.007 | 0.019 | 1.000 | 0.049 | 0.075 | 0.016 | 0.005 | 0.015 | -0.032 | 0.002 | 0.020 | 0.105 | -0.040 |
| GUA | 0.091 | 0.037 | 0.051 | 0.060 | -0.048 | 0.022 | -0.014 | 0.074 | -0.002 | -0.045 | 0.004 | -0.073 | -0.035 | 0.049 | 1.000 | -0.035 | 0.041 | 0.013 | 0.069 | 0.022 | 0.068 | 0.015 | -0.007 | 0.063 |
| UB | 0.105 | -0.013 | 0.034 | 0.046 | 0.006 | 0.060 | 0.008 | 0.024 | 0.026 | 0.036 | 0.058 | 0.047 | 0.044 | 0.075 | -0.035 | 1.000 | 0.032 | -0.058 | -0.014 | 0.077 | 0.035 | -0.035 | 0.049 | -0.052 |
| H | 0.113 | -0.027 | 0.048 | 0.092 | 0.047 | 0.032 | -0.003 | 0.015 | 0.080 | 0.033 | 0.022 | -0.048 | 0.032 | 0.016 | 0.041 | 0.032 | 1.000 | 0.037 | 0.031 | -0.020 | 0.036 | -0.040 | -0.088 | 0.040 |
| DEXE | 0.029 | -0.079 | -0.026 | 0.016 | 0.064 | 0.011 | 0.046 | 0.118 | 0.035 | -0.013 | -0.010 | -0.020 | 0.045 | 0.005 | 0.013 | -0.058 | 0.037 | 1.000 | 0.013 | 0.031 | 0.005 | 0.011 | 0.019 | 0.027 |
| BEL | 0.158 | -0.021 | 0.017 | -0.008 | 0.086 | 0.054 | -0.001 | 0.026 | 0.033 | 0.004 | 0.026 | 0.004 | 0.007 | 0.015 | 0.069 | -0.014 | 0.031 | 0.013 | 1.000 | -0.028 | 0.043 | -0.013 | 0.079 | -0.078 |
| GPS | 0.063 | -0.003 | 0.013 | -0.061 | 0.083 | -0.004 | -0.038 | 0.028 | 0.011 | 0.026 | 0.061 | -0.033 | 0.033 | -0.032 | 0.022 | 0.077 | -0.020 | 0.031 | -0.028 | 1.000 | 0.022 | -0.050 | 0.014 | 0.036 |
| BABY | 0.109 | 0.015 | 0.013 | -0.061 | 0.035 | 0.010 | 0.033 | 0.041 | -0.043 | -0.030 | 0.060 | -0.087 | 0.016 | 0.002 | 0.068 | 0.035 | 0.036 | 0.005 | 0.043 | 0.022 | 1.000 | -0.007 | 0.011 | -0.054 |
| STABLE | 0.058 | 0.082 | -0.018 | -0.052 | -0.016 | -0.023 | 0.018 | 0.026 | 0.047 | -0.079 | -0.010 | -0.002 | 0.062 | 0.020 | 0.015 | -0.035 | -0.040 | 0.011 | -0.013 | -0.050 | -0.007 | 1.000 | 0.034 | -0.050 |
| DODO | 0.069 | -0.032 | -0.021 | 0.028 | -0.055 | -0.020 | -0.025 | -0.007 | 0.040 | 0.009 | -0.065 | 0.015 | 0.026 | 0.105 | -0.007 | 0.049 | -0.088 | 0.019 | 0.079 | 0.014 | 0.011 | 0.034 | 1.000 | 0.001 |
| XPIN | 0.045 | 0.009 | -0.055 | -0.008 | 0.041 | 0.017 | -0.093 | 0.050 | 0.035 | -0.043 | 0.007 | -0.052 | -0.025 | -0.040 | 0.063 | -0.052 | 0.040 | 0.027 | -0.078 | 0.036 | -0.054 | -0.050 | 0.001 | 1.000 |

The complete 182 × 182 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| ASR | ASRUSDT | 60.5% | 62.9% | PSGUSDT | 0.364 |
| COPPER | COPPERUSDT | 60.5% | 62.9% | BTCUSDT | 0.303 |
| TUT | TUTUSDT | 60.5% | 62.8% | BTCUSDT | 0.376 |
| ARIA | ARIAUSDT | 61.0% | 62.4% | BTCUSDT | 0.329 |
| MEGA | MEGAUSDT | 61.0% | 62.4% | BTCUSDT | 0.391 |
| RAVE | RAVEUSDT | 61.3% | 62.2% | ACTUSDT | 0.369 |
| BASED | BASEDUSDT | 61.3% | 62.2% | TAIKOUSDT | 0.287 |
| TRUST | TRUSTUSDT | 61.4% | 62.1% | BTCUSDT | 0.480 |
| KOMA | KOMAUSDT | 61.5% | 62.1% | ZAMAUSDT | 0.271 |
| POWR | POWRUSDT | 61.6% | 62.0% | BTCUSDT | 0.451 |
| AVAAI | AVAAIUSDT | 61.7% | 61.9% | BTCUSDT | 0.294 |
| MMT | MMTUSDT | 61.7% | 61.9% | ARKUSDT | 0.245 |
| EWJ | EWJUSDT | 61.7% | 61.9% | FLNCUSDT | 0.352 |
| ONG | ONGUSDT | 61.8% | 61.8% | BTCUSDT | 0.359 |
| SAFE | SAFEUSDT | 61.8% | 61.8% | BTCUSDT | 0.438 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

