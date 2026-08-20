# Binance portfolio basis

Generated 2026-07-23T19:17:05.254Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-05-24T00:00Z through 2026-07-22T20:00Z
- Sampling: exactly 360 4h log returns (60 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 5.0% of the maximum unexplained variance, select the largest mean absolute 4h return
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
| 4 | H | HUSDT (usdm-futures) | usdm-futures | 99.2% | 0.113 | BTCUSDT | 579.76 bp | 100.0% | 0 | 46.8M | 574.4% |
| 5 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 99.5% | 0.092 | HUSDT | 571.67 bp | 100.0% | 0 | 45.5M | 482.9% |
| 6 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 99.2% | 0.091 | BTCUSDT | 523.71 bp | 100.0% | 4 | 18.2M | 441.6% |
| 7 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 97.7% | 0.149 | VELVETUSDT | 504.44 bp | 100.0% | 0 | 71.1M | 339.3% |
| 8 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 98.2% | 0.103 | HUSDT | 489.14 bp | 100.0% | 0 | 22.6M | 350.8% |
| 9 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 98.4% | 0.144 | CLOUSDT | 469.39 bp | 100.0% | 4 | 10.6M | 363.4% |
| 10 | SYN | SYNUSDT (spot) | spot, usdm-futures | 98.3% | 0.084 | EVAAUSDT | 458.82 bp | 100.0% | 4 | 6.2M | 326.5% |
| 11 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 97.6% | 0.110 | ESPORTSUSDT | 443.90 bp | 100.0% | 0 | 32.9M | 319.1% |
| 12 | HEI | HEIUSDT (spot) | spot, usdm-futures | 98.9% | 0.097 | SYNUSDT | 409.08 bp | 100.0% | 4 | 3.9M | 329.6% |
| 13 | UB | UBUSDT (usdm-futures) | usdm-futures | 98.1% | 0.105 | BTCUSDT | 403.73 bp | 100.0% | 4 | 27.5M | 271.9% |
| 14 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 99.0% | 0.126 | BEATUSDT | 401.62 bp | 100.0% | 4 | 10.2M | 313.4% |
| 15 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 97.5% | 0.131 | SKYAIUSDT | 400.00 bp | 100.0% | 0 | 27.3M | 279.0% |
| 16 | RIF | RIFUSDT (spot) | spot, usdm-futures | 98.7% | 0.089 | BTCUSDT | 385.12 bp | 100.0% | 4 | 2M | 273.7% |
| 17 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 98.1% | 0.101 | EVAAUSDT | 375.27 bp | 100.0% | 0 | 11.4M | 291.3% |
| 18 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 98.5% | 0.080 | HUSDT | 370.24 bp | 100.0% | 4 | 3.8M | 280.9% |
| 19 | US | USUSDT (usdm-futures) | usdm-futures | 97.1% | 0.107 | CLOUSDT | 368.84 bp | 100.0% | 4 | 8.6M | 272.4% |
| 20 | BAS | BASUSDT (usdm-futures) | usdm-futures | 98.4% | 0.090 | BTCUSDT | 351.26 bp | 100.0% | 0 | 7.8M | 268.4% |
| 21 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 97.1% | 0.109 | EVAAUSDT | 333.22 bp | 100.0% | 0 | 3.6M | 253.5% |
| 22 | BANK | BANKUSDT (spot) | spot, usdm-futures | 97.7% | 0.141 | BTCUSDT | 319.04 bp | 100.0% | 8 | 511.5K | 338.0% |
| 23 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 97.0% | 0.110 | HOMEUSDT | 318.64 bp | 100.0% | 4 | 12.4M | 230.2% |
| 24 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 97.2% | 0.117 | SKYAIUSDT | 316.45 bp | 100.0% | 4 | 2.1M | 253.0% |
| 25 | M | MUSDT (usdm-futures) | usdm-futures | 98.8% | 0.059 | SYNUSDT | 306.18 bp | 100.0% | 4 | 4.5M | 400.9% |
| 26 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 95.6% | 0.114 | GUAUSDT | 369.33 bp | 100.0% | 8 | 3.1M | 276.4% |
| 27 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 95.8% | 0.147 | BTCUSDT | 360.74 bp | 100.0% | 0 | 9.9M | 286.5% |
| 28 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 95.8% | 0.138 | ESPORTSUSDT | 328.13 bp | 100.0% | 0 | 4.9M | 233.9% |
| 29 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 95.8% | 0.132 | USUSDT | 313.58 bp | 100.0% | 4 | 1.8M | 350.6% |
| 30 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 95.5% | 0.155 | BTCUSDT | 302.46 bp | 100.0% | 4 | 1.5M | 317.6% |
| 31 | B | BUSDT (usdm-futures) | usdm-futures | 97.5% | 0.106 | BANKUSDT | 288.56 bp | 100.0% | 4 | 4.9M | 281.3% |
| 32 | IN | INUSDT (usdm-futures) | usdm-futures | 97.0% | 0.127 | GWEIUSDT | 279.77 bp | 100.0% | 0 | 6.9M | 366.0% |
| 33 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 95.5% | 0.092 | HOMEUSDT | 257.53 bp | 100.0% | 4 | 4.5M | 174.9% |
| 34 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 96.0% | 0.157 | VELVETUSDT | 242.17 bp | 100.0% | 8 | 1.2M | 400.6% |
| 35 | BEL | BELUSDT (spot) | spot, usdm-futures | 96.2% | 0.158 | BTCUSDT | 241.93 bp | 100.0% | 4 | 723.1K | 198.7% |
| 36 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 95.5% | 0.108 | BTCUSDT | 227.10 bp | 100.0% | 4 | 2.3M | 174.7% |
| 37 | ATM | ATMUSDT (spot) | spot | 95.6% | 0.129 | BASUSDT | 226.41 bp | 100.0% | 4 | 971.8K | 202.5% |
| 38 | DODO | DODOUSDT (spot) | spot | 96.2% | 0.105 | INUSDT | 213.61 bp | 100.0% | 4 | 623.1K | 174.2% |
| 39 | BABY | BABYUSDT (spot) | spot, usdm-futures | 96.3% | 0.109 | BTCUSDT | 164.57 bp | 100.0% | 4 | 637K | 184.6% |
| 40 | TA | TAUSDT (usdm-futures) | usdm-futures | 94.6% | 0.135 | RIFUSDT | 175.80 bp | 100.0% | 4 | 2.3M | 133.8% |
| 41 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 94.3% | 0.154 | BTCUSDT | 329.57 bp | 100.0% | 0 | 3.8M | 249.0% |
| 42 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 93.9% | 0.197 | UAIUSDT | 226.45 bp | 100.0% | 0 | 3.6M | 162.5% |
| 43 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 94.7% | 0.113 | RIFUSDT | 171.95 bp | 100.0% | 4 | 687.1K | 182.6% |
| 44 | XNO | XNOUSDT (spot) | spot | 94.1% | 0.226 | BTCUSDT | 161.08 bp | 100.0% | 8 | 61.1K | 138.7% |
| 45 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 94.1% | 0.117 | CLOUSDT | 157.74 bp | 100.0% | 4 | 2.6M | 108.7% |
| 46 | Q | QUSDT (usdm-futures) | usdm-futures | 93.8% | 0.158 | TAUSDT | 155.95 bp | 100.0% | 4 | 1.9M | 110.5% |
| 47 | CATI | CATIUSDT (spot) | spot, usdm-futures | 93.5% | 0.165 | BTCUSDT | 155.82 bp | 100.0% | 12 | 408.3K | 117.3% |
| 48 | B2 | B2USDT (usdm-futures) | usdm-futures | 93.3% | 0.122 | INUSDT | 146.86 bp | 100.0% | 4 | 1.4M | 101.1% |
| 49 | PIVX | PIVXUSDT (spot) | spot | 93.1% | 0.207 | VELVETUSDT | 216.91 bp | 100.0% | 8 | 205K | 245.8% |
| 50 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 92.9% | 0.186 | BTCUSDT | 167.94 bp | 100.0% | 0 | 1.2M | 148.1% |
| 51 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 93.8% | 0.127 | SKYAIUSDT | 140.41 bp | 100.0% | 4 | 3.7M | 98.4% |
| 52 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 95.2% | 0.125 | ESPORTSUSDT | 135.63 bp | 100.0% | 4 | 1.5M | 91.3% |
| 53 | STG | STGUSDT (spot) | spot, usdm-futures | 92.0% | 0.141 | 龙虾USDT | 304.22 bp | 100.0% | 4 | 1.5M | 244.6% |
| 54 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 91.7% | 0.143 | BTCUSDT | 302.56 bp | 100.0% | 4 | 13.6M | 216.2% |
| 55 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 91.8% | 0.168 | ALLOUSDT | 252.52 bp | 100.0% | 0 | 4.2M | 195.5% |
| 56 | APR | APRUSDT (usdm-futures) | usdm-futures | 91.5% | 0.143 | MAGMAUSDT | 199.38 bp | 100.0% | 4 | 3.3M | 138.5% |
| 57 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 93.3% | 0.193 | BANKUSDT | 134.41 bp | 100.0% | 8 | 1.1M | 123.4% |
| 58 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 91.7% | 0.207 | PORTALUSDT | 55.08 bp | 100.0% | 4 | 20.6M | 41.1% |
| 59 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 91.9% | 0.130 | BTCUSDT | 43.74 bp | 100.0% | 0 | 6.4M | 34.6% |
| 60 | KGST | KGSTUSDT (spot) | spot | 93.9% | 0.110 | BABYUSDT | 5.59 bp | 100.0% | 24 | 75.5K | 4.5% |
| 61 | TLM | TLMUSDT (spot) | spot, usdm-futures | 90.9% | 0.203 | BTCUSDT | 323.65 bp | 100.0% | 8 | 793.1K | 294.4% |
| 62 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 89.3% | 0.249 | LABUSDT | 301.54 bp | 100.0% | 4 | 2.9M | 324.4% |
| 63 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 89.1% | 0.174 | BTCUSDT | 286.62 bp | 100.0% | 4 | 4.1M | 196.8% |
| 64 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 89.6% | 0.210 | BTCUSDT | 269.38 bp | 100.0% | 4 | 13.1M | 209.5% |
| 65 | BICO | BICOUSDT (spot) | spot, usdm-futures | 90.3% | 0.251 | BTCUSDT | 226.15 bp | 100.0% | 16 | 629.4K | 205.7% |
| 66 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 89.2% | 0.188 | BTCUSDT | 220.89 bp | 100.0% | 4 | 772.5K | 184.9% |
| 67 | STRAX | STRAXUSDT (spot) | spot | 90.6% | 0.207 | UAIUSDT | 191.29 bp | 100.0% | 4 | 680K | 194.2% |
| 68 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 89.0% | 0.207 | BTCUSDT | 183.65 bp | 100.0% | 4 | 2.1M | 135.1% |
| 69 | DRIFT | DRIFTUSDT (usdm-futures) | usdm-futures | 89.8% | 0.263 | BTCUSDT | 182.37 bp | 100.0% | 4 | 1.9M | 155.7% |
| 70 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 89.7% | 0.287 | BTCUSDT | 178.21 bp | 100.0% | 4 | 1.3M | 139.9% |
| 71 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 89.3% | 0.352 | BTCUSDT | 167.48 bp | 100.0% | 4 | 1.9M | 130.0% |
| 72 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 89.0% | 0.231 | BTCUSDT | 166.50 bp | 100.0% | 8 | 592K | 154.9% |
| 73 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 88.0% | 0.240 | BTCUSDT | 312.78 bp | 100.0% | 4 | 4.3M | 229.8% |
| 74 | STAR | STARUSDT (usdm-futures) | usdm-futures | 87.9% | 0.202 | BTCUSDT | 246.74 bp | 100.0% | 4 | 1.8M | 164.7% |
| 75 | ID | IDUSDT (spot) | spot, usdm-futures | 87.1% | 0.166 | STGUSDT | 244.93 bp | 100.0% | 12 | 1.5M | 174.5% |
| 76 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 87.6% | 0.153 | TRIAUSDT | 220.99 bp | 100.0% | 4 | 1.7M | 156.5% |
| 77 | ACE | ACEUSDT (spot) | spot, usdm-futures | 88.3% | 0.237 | BTCUSDT | 163.81 bp | 100.0% | 8 | 359K | 154.7% |
| 78 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 86.8% | 0.225 | BTCUSDT | 200.47 bp | 100.0% | 4 | 3.2M | 156.1% |
| 79 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 87.2% | 0.200 | BTCUSDT | 157.00 bp | 100.0% | 4 | 3.7M | 103.7% |
| 80 | DGB | DGBUSDT (spot) | spot | 87.2% | 0.280 | XNOUSDT | 154.86 bp | 100.0% | 8 | 105.3K | 120.7% |
| 81 | XEC | XECUSDT (spot) | spot, usdm-futures | 88.9% | 0.267 | BTCUSDT | 146.21 bp | 100.0% | 8 | 187.6K | 124.1% |
| 82 | FF | FFUSDT (spot) | spot, usdm-futures | 87.2% | 0.208 | BTCUSDT | 132.36 bp | 100.0% | 4 | 1.6M | 98.2% |
| 83 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 86.2% | 0.223 | VELVETUSDT | 303.14 bp | 100.0% | 4 | 8.9M | 218.3% |
| 84 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 85.5% | 0.372 | BANKUSDT | 302.43 bp | 100.0% | 0 | 4.8M | 339.0% |
| 85 | OPG | OPGUSDT (spot) | spot, usdm-futures | 85.5% | 0.208 | BTCUSDT | 253.21 bp | 100.0% | 4 | 4.3M | 191.2% |
| 86 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 85.2% | 0.135 | FFUSDT | 229.28 bp | 100.0% | 0 | 2.7M | 160.7% |
| 87 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 85.7% | 0.235 | BTCUSDT | 227.84 bp | 100.0% | 0 | 269.4K | 207.7% |
| 88 | ACT | ACTUSDT (spot) | spot, usdm-futures | 85.9% | 0.273 | BTCUSDT | 194.45 bp | 100.0% | 4 | 504K | 184.0% |
| 89 | OSMO | OSMOUSDT (spot) | spot | 85.1% | 0.223 | BTCUSDT | 179.16 bp | 100.0% | 8 | 466.3K | 147.0% |
| 90 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 84.7% | 0.279 | BTCUSDT | 157.43 bp | 100.0% | 8 | 764K | 151.8% |
| 91 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 84.6% | 0.206 | POWERUSDT | 244.31 bp | 100.0% | 4 | 4.3M | 172.6% |
| 92 | PORTO | PORTOUSDT (spot) | spot | 84.6% | 0.236 | AKEUSDT | 140.71 bp | 100.0% | 4 | 302.6K | 128.1% |
| 93 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 84.3% | 0.155 | BTCUSDT | 169.02 bp | 100.0% | 4 | 1.4M | 114.4% |
| 94 | TAC | TACUSDT (usdm-futures) | usdm-futures | 84.1% | 0.233 | LABUSDT | 381.48 bp | 100.0% | 4 | 4.1M | 616.4% |
| 95 | REQ | REQUSDT (spot) | spot | 83.8% | 0.309 | BTCUSDT | 125.44 bp | 100.0% | 12 | 118.1K | 114.3% |
| 96 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 83.8% | 0.246 | ESPORTSUSDT | 358.00 bp | 100.0% | 4 | 16M | 335.7% |
| 97 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 82.6% | 0.192 | TAUSDT | 216.27 bp | 100.0% | 8 | 1.6M | 152.4% |
| 98 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 82.3% | 0.237 | BANKUSDT | 246.90 bp | 100.0% | 4 | 1.8M | 188.1% |
| 99 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 83.1% | 0.211 | OPGUSDT | 197.74 bp | 100.0% | 4 | 2.3M | 143.6% |
| 100 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 82.6% | 0.293 | TAGUSDT | 190.80 bp | 100.0% | 4 | 1.8M | 146.0% |
| 101 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 82.3% | 0.226 | DRIFTUSDT | 189.29 bp | 100.0% | 4 | 3.2M | 135.3% |
| 102 | QUICK | QUICKUSDT (spot) | spot | 83.1% | 0.211 | PIVXUSDT | 177.02 bp | 100.0% | 4 | 204.1K | 155.7% |
| 103 | G | GUSDT (spot) | spot, usdm-futures | 82.5% | 0.344 | BTCUSDT | 155.94 bp | 100.0% | 12 | 326.1K | 116.1% |
| 104 | PROM | PROMUSDT (spot) | spot, usdm-futures | 83.0% | 0.331 | BANKUSDT | 149.68 bp | 100.0% | 8 | 276K | 144.1% |
| 105 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 84.2% | 0.298 | BTCUSDT | 123.93 bp | 100.0% | 4 | 178.2K | 138.6% |
| 106 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 81.4% | 0.192 | BABYUSDT | 177.05 bp | 100.0% | 4 | 1.2M | 131.8% |
| 107 | FLNC | FLNCUSDT (usdm-futures) | usdm-futures | 81.3% | 0.213 | BTCUSDT | 172.90 bp | 100.0% | 4 | 5.2M | 135.6% |
| 108 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 80.8% | 0.234 | ESPORTSUSDT | 230.57 bp | 100.0% | 0 | 3.4M | 217.3% |
| 109 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 81.2% | 0.212 | MANTAUSDT | 170.03 bp | 100.0% | 0 | 5.2M | 145.3% |
| 110 | RPL | RPLUSDT (spot) | spot, usdm-futures | 81.7% | 0.342 | BTCUSDT | 158.51 bp | 100.0% | 16 | 125.3K | 136.0% |
| 111 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 80.7% | 0.170 | PORTOUSDT | 155.30 bp | 100.0% | 4 | 1.1M | 102.7% |
| 112 | BAN | BANUSDT (usdm-futures) | usdm-futures | 80.3% | 0.178 | GENIUSUSDT | 170.64 bp | 100.0% | 0 | 2.6M | 116.7% |
| 113 | QKC | QKCUSDT (spot) | spot | 82.3% | 0.341 | BTCUSDT | 109.09 bp | 100.0% | 4 | 93.6K | 96.2% |
| 114 | XAN | XANUSDT (usdm-futures) | usdm-futures | 80.1% | 0.197 | BTCUSDT | 235.07 bp | 100.0% | 4 | 2.8M | 157.9% |
| 115 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 79.6% | 0.195 | BTCUSDT | 194.28 bp | 100.0% | 0 | 1.5M | 130.4% |
| 116 | SXT | SXTUSDT (spot) | spot, usdm-futures | 78.8% | 0.310 | BTCUSDT | 186.99 bp | 100.0% | 4 | 618.4K | 140.4% |
| 117 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 78.7% | 0.253 | BTCUSDT | 171.61 bp | 100.0% | 4 | 2.9M | 159.9% |
| 118 | SKL | SKLUSDT (spot) | spot, usdm-futures | 78.5% | 0.374 | BTCUSDT | 155.85 bp | 100.0% | 8 | 410.9K | 126.9% |
| 119 | ERA | ERAUSDT (spot) | spot, usdm-futures | 78.8% | 0.284 | BTCUSDT | 154.07 bp | 100.0% | 4 | 340.3K | 168.0% |
| 120 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 79.1% | 0.326 | BTCUSDT | 128.31 bp | 100.0% | 0 | 35.1M | 88.1% |
| 121 | USTC | USTCUSDT (spot) | spot, usdm-futures | 78.2% | 0.385 | BTCUSDT | 111.52 bp | 100.0% | 4 | 367K | 78.6% |
| 122 | CITY | CITYUSDT (spot) | spot | 78.4% | 0.258 | BTCUSDT | 107.83 bp | 100.0% | 8 | 440K | 95.3% |
| 123 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 77.7% | 0.204 | BTCUSDT | 227.90 bp | 100.0% | 0 | 1.9M | 154.9% |
| 124 | CL | CLUSDT (usdm-futures) | usdm-futures | 79.1% | 0.179 | NAORISUSDT | 74.24 bp | 100.0% | 4 | 474.5M | 48.5% |
| 125 | V | VUSDT (usdm-futures) | usdm-futures | 78.3% | 0.194 | AMZNUSDT | 39.84 bp | 100.0% | 4 | 301.7K | 29.1% |
| 126 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 77.5% | 0.153 | MANTAUSDT | 26.40 bp | 100.0% | 4 | 444.2K | 19.1% |
| 127 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 76.8% | 0.277 | BTCUSDT | 234.07 bp | 100.0% | 8 | 922.1K | 172.6% |
| 128 | MITO | MITOUSDT (spot) | spot, usdm-futures | 76.7% | 0.183 | COAIUSDT | 197.71 bp | 100.0% | 4 | 920.7K | 145.6% |
| 129 | U | UUSDT (spot) | spot | 78.3% | 0.157 | FLNCUSDT | 0.91 bp | 100.0% | 24 | 15.9M | 0.6% |
| 130 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 74.9% | 0.171 | ESPORTSUSDT | 258.87 bp | 100.0% | 0 | 4.1M | 171.6% |
| 131 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 75.3% | 0.321 | BTCUSDT | 256.09 bp | 100.0% | 4 | 2.4M | 165.9% |
| 132 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 74.1% | 0.319 | BASUSDT | 232.61 bp | 100.0% | 4 | 2.3M | 172.1% |
| 133 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 74.7% | 0.244 | TRADOORUSDT | 209.71 bp | 100.0% | 0 | 1.1M | 141.3% |
| 134 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 75.1% | 0.259 | FFUSDT | 181.88 bp | 100.0% | 4 | 1.9M | 236.6% |
| 135 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 74.4% | 0.340 | BTCUSDT | 178.17 bp | 100.0% | 4 | 1.9M | 135.3% |
| 136 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 74.2% | 0.295 | PROMUSDT | 160.86 bp | 100.0% | 4 | 2.9M | 100.4% |
| 137 | KITE | KITEUSDT (spot) | spot, usdm-futures | 75.7% | 0.301 | BTCUSDT | 157.74 bp | 100.0% | 4 | 2.5M | 100.7% |
| 138 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 73.2% | 0.288 | BTCUSDT | 181.62 bp | 100.0% | 4 | 2.5M | 152.0% |
| 139 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 73.0% | 0.282 | SXTUSDT | 180.36 bp | 100.0% | 4 | 3.8M | 138.7% |
| 140 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 72.9% | 0.195 | TRADOORUSDT | 158.76 bp | 100.0% | 4 | 3.3M | 121.4% |
| 141 | GPS | GPSUSDT (spot) | spot, usdm-futures | 73.6% | 0.228 | MYXUSDT | 153.70 bp | 100.0% | 4 | 691.7K | 127.5% |
| 142 | AT | ATUSDT (spot) | spot, usdm-futures | 73.1% | 0.186 | NAORISUSDT | 145.09 bp | 100.0% | 4 | 356.4K | 111.6% |
| 143 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 72.5% | 0.365 | BTCUSDT | 146.87 bp | 100.0% | 4 | 424.1K | 102.4% |
| 144 | T | TUSDT (spot) | spot, usdm-futures | 74.3% | 0.356 | BTCUSDT | 139.06 bp | 100.0% | 8 | 234.3K | 115.3% |
| 145 | ZORA | ZORAUSDT (usdm-futures) | usdm-futures | 72.2% | 0.350 | BTCUSDT | 149.47 bp | 100.0% | 4 | 2.4M | 103.4% |
| 146 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 71.5% | 0.306 | BTCUSDT | 132.04 bp | 100.0% | 4 | 663.8K | 81.5% |
| 147 | AWE | AWEUSDT (spot) | spot, usdm-futures | 72.8% | 0.211 | GUSDT | 125.14 bp | 100.0% | 4 | 381.2K | 82.7% |
| 148 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 71.4% | 0.217 | BTCUSDT | 124.10 bp | 100.0% | 4 | 827.1K | 90.6% |
| 149 | SKY | SKYUSDT (spot) | spot, usdm-futures | 72.0% | 0.437 | BTCUSDT | 102.03 bp | 100.0% | 4 | 1.2M | 66.4% |
| 150 | JST | JSTUSDT (spot) | spot, usdm-futures | 72.9% | 0.186 | BRKBUSDT | 90.19 bp | 100.0% | 4 | 3.5M | 63.6% |
| 151 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 69.4% | 0.292 | COAIUSDT | 273.61 bp | 100.0% | 4 | 2.6M | 198.6% |
| 152 | BR | BRUSDT (usdm-futures) | usdm-futures | 70.1% | 0.230 | BTCUSDT | 232.19 bp | 100.0% | 0 | 1.8M | 173.3% |
| 153 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 69.4% | 0.317 | TNSRUSDT | 218.28 bp | 100.0% | 4 | 2.5M | 157.5% |
| 154 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 69.7% | 0.328 | COAIUSDT | 187.21 bp | 100.0% | 4 | 3.5M | 164.8% |
| 155 | PYR | PYRUSDT (spot) | spot | 69.9% | 0.321 | XPINUSDT | 175.86 bp | 100.0% | 16 | 662.6K | 141.7% |
| 156 | XLM | XLMUSDT (spot) | spot, usdm-futures, coinm-futures | 69.5% | 0.374 | BTCUSDT | 167.62 bp | 100.0% | 12 | 19.3M | 123.6% |
| 157 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 69.9% | 0.276 | PORTOUSDT | 148.64 bp | 100.0% | 4 | 2.4M | 112.1% |
| 158 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 69.1% | 0.206 | TRADOORUSDT | 146.14 bp | 100.0% | 0 | 2.5M | 107.1% |
| 159 | DCR | DCRUSDT (spot) | spot | 70.2% | 0.371 | XECUSDT | 110.30 bp | 100.0% | 4 | 200.1K | 86.0% |
| 160 | HD | HDUSDT (usdm-futures) | usdm-futures | 69.5% | 0.339 | CLUSDT | 40.80 bp | 100.0% | 4 | 395.1K | 32.0% |
| 161 | WLD | WLDUSDT (spot) | spot, usdm-futures | 67.7% | 0.358 | BTCUSDT | 242.47 bp | 100.0% | 0 | 43.1M | 176.7% |
| 162 | ON | ONUSDT (usdm-futures) | usdm-futures | 68.1% | 0.343 | PROMUSDT | 233.78 bp | 100.0% | 4 | 1.5M | 168.7% |
| 163 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 68.3% | 0.373 | FIDAUSDT | 218.99 bp | 100.0% | 4 | 1.9M | 138.6% |
| 164 | THE | THEUSDT (spot) | spot, usdm-futures | 67.6% | 0.361 | RPLUSDT | 171.59 bp | 100.0% | 8 | 546.5K | 146.3% |
| 165 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 68.1% | 0.268 | BTCUSDT | 86.89 bp | 100.0% | 8 | 559K | 71.0% |
| 166 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 68.1% | 0.279 | FLNCUSDT | 48.23 bp | 100.0% | 4 | 544.6K | 40.9% |
| 167 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 67.0% | 0.230 | VUSDT | 39.49 bp | 100.0% | 4 | 414.6K | 31.4% |
| 168 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 67.7% | 0.366 | AMZNUSDT | 38.53 bp | 100.0% | 4 | 9.3M | 31.5% |
| 169 | DIS | DISUSDT (usdm-futures) | usdm-futures | 68.1% | 0.319 | HDUSDT | 36.80 bp | 100.0% | 4 | 272K | 27.3% |
| 170 | JTO | JTOUSDT (spot) | spot, usdm-futures | 65.3% | 0.330 | BTCUSDT | 239.05 bp | 100.0% | 0 | 5.8M | 159.3% |
| 171 | VIC | VICUSDT (spot) | spot, usdm-futures | 64.5% | 0.259 | PORTALUSDT | 221.87 bp | 100.0% | 12 | 972.2K | 191.6% |
| 172 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 65.6% | 0.294 | BTCUSDT | 205.23 bp | 100.0% | 4 | 1.2M | 164.8% |
| 173 | LIT | LITUSDT (usdm-futures) | usdm-futures | 64.0% | 0.407 | BTCUSDT | 256.00 bp | 100.0% | 0 | 50.9M | 162.8% |
| 174 | AIN | AINUSDT (usdm-futures) | usdm-futures | 63.9% | 0.227 | SWARMSUSDT | 250.32 bp | 100.0% | 0 | 1.3M | 181.5% |
| 175 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 65.3% | 0.367 | TRADOORUSDT | 194.41 bp | 100.0% | 4 | 2.6M | 126.9% |
| 176 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 63.1% | 0.329 | BTCUSDT | 205.02 bp | 100.0% | 8 | 1.9M | 144.5% |
| 177 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 63.0% | 0.223 | FOLKSUSDT | 269.61 bp | 100.0% | 0 | 2.5M | 211.9% |
| 178 | SENT | SENTUSDT (spot) | spot, usdm-futures | 63.7% | 0.262 | ALCHUSDT | 179.60 bp | 100.0% | 4 | 1.1M | 120.8% |
| 179 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 63.0% | 0.438 | BTCUSDT | 160.21 bp | 100.0% | 4 | 760.9K | 107.2% |
| 180 | MMT | MMTUSDT (spot) | spot, usdm-futures | 62.7% | 0.234 | PYRUSDT | 185.75 bp | 100.0% | 4 | 851.1K | 127.7% |
| 181 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 63.6% | 0.271 | ZAMAUSDT | 159.00 bp | 100.0% | 4 | 944.8K | 114.0% |
| 182 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 63.2% | 0.375 | BTCUSDT | 147.43 bp | 100.0% | 4 | 1.9M | 93.3% |

## Diagnostics

- Basis size selected: 182
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.064
- Maximum pairwise absolute correlation: 0.438
- Mean whole-market projection R²: 82.4%
- Median whole-market projection R²: 80.2%
- 10th-percentile whole-market projection R²: 64.9%
- Minimum whole-market projection R²: 59.1%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.1% | 20.3% | 1.1% | 0.0% |
| 5 | 24.7% | 21.1% | 2.6% | 0.2% |
| 10 | 26.7% | 22.9% | 4.7% | 1.0% |
| 15 | 28.6% | 24.4% | 6.1% | 1.4% |
| 20 | 30.6% | 26.6% | 7.6% | 1.6% |
| 25 | 32.5% | 28.3% | 9.2% | 4.2% |
| 30 | 34.5% | 30.6% | 11.1% | 4.7% |
| 35 | 37.0% | 33.4% | 13.1% | 5.0% |
| 40 | 39.5% | 35.6% | 15.0% | 7.1% |
| 45 | 42.0% | 37.9% | 17.4% | 7.4% |
| 50 | 43.9% | 40.0% | 19.0% | 9.4% |
| 55 | 45.8% | 42.1% | 20.3% | 11.9% |
| 60 | 47.5% | 43.3% | 22.4% | 16.1% |
| 65 | 49.3% | 45.0% | 23.8% | 17.0% |
| 70 | 51.1% | 46.9% | 25.6% | 17.1% |
| 75 | 53.0% | 49.0% | 27.4% | 20.5% |
| 80 | 55.2% | 51.7% | 29.4% | 21.0% |
| 85 | 57.1% | 53.5% | 31.8% | 23.6% |
| 90 | 59.0% | 55.6% | 33.7% | 25.1% |
| 95 | 60.5% | 57.1% | 35.3% | 26.7% |
| 100 | 62.0% | 58.5% | 37.2% | 29.0% |
| 105 | 63.6% | 59.5% | 39.4% | 30.5% |
| 110 | 65.1% | 61.1% | 41.1% | 31.7% |
| 115 | 66.6% | 63.1% | 43.2% | 35.1% |
| 120 | 68.1% | 64.4% | 45.0% | 35.7% |
| 125 | 69.6% | 66.1% | 46.9% | 37.5% |
| 130 | 71.0% | 67.6% | 48.4% | 42.1% |
| 135 | 72.3% | 69.1% | 50.6% | 42.6% |
| 140 | 73.5% | 70.0% | 52.3% | 44.6% |
| 145 | 74.8% | 71.5% | 54.2% | 46.5% |
| 150 | 75.9% | 72.6% | 55.6% | 49.5% |
| 155 | 76.8% | 73.6% | 57.1% | 49.7% |
| 160 | 77.9% | 74.7% | 58.4% | 52.2% |
| 165 | 79.1% | 76.7% | 59.5% | 52.7% |
| 170 | 80.1% | 77.9% | 61.2% | 56.7% |
| 175 | 81.1% | 78.9% | 63.2% | 58.1% |
| 180 | 82.0% | 79.7% | 64.5% | 59.0% |
| 182 | 82.4% | 80.2% | 64.9% | 59.1% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | ESPORTS | LAB | H | VELVET | GUA | BEAT | CLO | EVAA | SYN | SKYAI | HEI | UB | ALLO | BSB | RIF | MAGMA | HOME | US | BAS | AGT | BANK | GWEI | HMSTR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.066 | 0.016 | 0.113 | 0.004 | 0.091 | -0.006 | 0.094 | 0.083 | -0.054 | 0.026 | 0.037 | 0.105 | 0.032 | 0.097 | 0.089 | 0.003 | 0.069 | 0.036 | 0.090 | 0.053 | 0.141 | 0.011 | 0.107 |
| ESPORTS | 0.066 | 1.000 | -0.005 | -0.027 | -0.029 | 0.037 | -0.022 | -0.066 | -0.029 | -0.073 | 0.110 | 0.019 | -0.013 | 0.030 | 0.099 | -0.028 | 0.042 | 0.005 | 0.058 | 0.014 | 0.010 | 0.033 | 0.030 | 0.032 |
| LAB | 0.016 | -0.005 | 1.000 | 0.048 | 0.032 | 0.051 | 0.074 | -0.046 | -0.022 | 0.043 | 0.098 | 0.023 | 0.034 | -0.021 | 0.056 | -0.010 | -0.027 | -0.008 | -0.057 | 0.005 | 0.031 | 0.011 | 0.041 | 0.034 |
| H | 0.113 | -0.027 | 0.048 | 1.000 | 0.092 | 0.041 | 0.090 | 0.103 | 0.047 | -0.040 | 0.064 | 0.032 | 0.032 | -0.003 | 0.017 | 0.015 | -0.026 | 0.080 | -0.048 | 0.022 | -0.049 | 0.016 | -0.087 | 0.045 |
| VELVET | 0.004 | -0.029 | 0.032 | 0.092 | 1.000 | 0.060 | 0.149 | -0.044 | -0.022 | -0.047 | 0.075 | 0.015 | 0.046 | -0.002 | 0.068 | 0.015 | 0.040 | -0.028 | -0.075 | 0.084 | 0.056 | 0.068 | -0.066 | 0.069 |
| GUA | 0.091 | 0.037 | 0.051 | 0.041 | 0.060 | 1.000 | -0.098 | 0.018 | -0.048 | 0.081 | 0.085 | 0.022 | -0.035 | -0.014 | 0.071 | 0.074 | 0.025 | -0.002 | -0.073 | 0.004 | 0.053 | 0.025 | -0.030 | -0.003 |
| BEAT | -0.006 | -0.022 | 0.074 | 0.090 | 0.149 | -0.098 | 1.000 | -0.091 | 0.014 | -0.074 | 0.101 | 0.041 | -0.024 | 0.126 | 0.015 | 0.028 | -0.006 | -0.014 | -0.048 | -0.035 | 0.030 | 0.047 | 0.068 | 0.041 |
| CLO | 0.094 | -0.066 | -0.046 | 0.103 | -0.044 | 0.018 | -0.091 | 1.000 | 0.144 | 0.009 | 0.016 | 0.076 | -0.065 | -0.006 | 0.043 | 0.026 | -0.058 | 0.064 | 0.107 | 0.010 | 0.070 | 0.009 | -0.029 | 0.073 |
| EVAA | 0.083 | -0.029 | -0.022 | 0.047 | -0.022 | -0.048 | 0.014 | 0.144 | 1.000 | 0.084 | 0.004 | 0.048 | 0.006 | 0.007 | 0.005 | -0.007 | -0.101 | 0.026 | 0.013 | 0.005 | -0.109 | 0.055 | 0.008 | -0.027 |
| SYN | -0.054 | -0.073 | 0.043 | -0.040 | -0.047 | 0.081 | -0.074 | 0.009 | 0.084 | 1.000 | 0.037 | 0.097 | 0.095 | -0.038 | 0.052 | -0.032 | 0.004 | -0.018 | 0.044 | -0.071 | 0.032 | -0.013 | -0.037 | -0.013 |
| SKYAI | 0.026 | 0.110 | 0.098 | 0.064 | 0.075 | 0.085 | 0.101 | 0.016 | 0.004 | 0.037 | 1.000 | 0.004 | -0.003 | 0.023 | 0.131 | 0.083 | 0.082 | 0.050 | 0.056 | 0.017 | -0.057 | 0.040 | 0.033 | 0.117 |
| HEI | 0.037 | 0.019 | 0.023 | 0.032 | 0.015 | 0.022 | 0.041 | 0.076 | 0.048 | 0.097 | 0.004 | 1.000 | 0.060 | 0.017 | -0.042 | 0.037 | -0.071 | 0.032 | -0.018 | 0.004 | 0.072 | 0.022 | -0.018 | 0.059 |
| UB | 0.105 | -0.013 | 0.034 | 0.032 | 0.046 | -0.035 | -0.024 | -0.065 | 0.006 | 0.095 | -0.003 | 0.060 | 1.000 | 0.008 | 0.030 | 0.024 | -0.062 | 0.026 | 0.047 | 0.058 | -0.051 | 0.003 | 0.016 | 0.037 |
| ALLO | 0.032 | 0.030 | -0.021 | -0.003 | -0.002 | -0.014 | 0.126 | -0.006 | 0.007 | -0.038 | 0.023 | 0.017 | 0.008 | 1.000 | 0.010 | 0.009 | 0.049 | 0.018 | 0.034 | 0.001 | 0.019 | 0.025 | 0.101 | 0.002 |
| BSB | 0.097 | 0.099 | 0.056 | 0.017 | 0.068 | 0.071 | 0.015 | 0.043 | 0.005 | 0.052 | 0.131 | -0.042 | 0.030 | 0.010 | 1.000 | 0.041 | 0.058 | -0.063 | -0.046 | -0.034 | 0.024 | 0.060 | 0.050 | 0.019 |
| RIF | 0.089 | -0.028 | -0.010 | 0.015 | 0.015 | 0.074 | 0.028 | 0.026 | -0.007 | -0.032 | 0.083 | 0.037 | 0.024 | 0.009 | 0.041 | 1.000 | -0.027 | 0.020 | -0.008 | 0.037 | -0.004 | 0.071 | 0.064 | -0.025 |
| MAGMA | 0.003 | 0.042 | -0.027 | -0.026 | 0.040 | 0.025 | -0.006 | -0.058 | -0.101 | 0.004 | 0.082 | -0.071 | -0.062 | 0.049 | 0.058 | -0.027 | 1.000 | 0.065 | 0.072 | -0.004 | 0.022 | -0.025 | 0.044 | 0.067 |
| HOME | 0.069 | 0.005 | -0.008 | 0.080 | -0.028 | -0.002 | -0.014 | 0.064 | 0.026 | -0.018 | 0.050 | 0.032 | 0.026 | 0.018 | -0.063 | 0.020 | 0.065 | 1.000 | -0.035 | 0.003 | -0.023 | 0.043 | 0.110 | 0.028 |
| US | 0.036 | 0.058 | -0.057 | -0.048 | -0.075 | -0.073 | -0.048 | 0.107 | 0.013 | 0.044 | 0.056 | -0.018 | 0.047 | 0.034 | -0.046 | -0.008 | 0.072 | -0.035 | 1.000 | -0.053 | 0.084 | 0.072 | 0.016 | 0.090 |
| BAS | 0.090 | 0.014 | 0.005 | 0.022 | 0.084 | 0.004 | -0.035 | 0.010 | 0.005 | -0.071 | 0.017 | 0.004 | 0.058 | 0.001 | -0.034 | 0.037 | -0.004 | 0.003 | -0.053 | 1.000 | -0.005 | 0.036 | -0.023 | 0.024 |
| AGT | 0.053 | 0.010 | 0.031 | -0.049 | 0.056 | 0.053 | 0.030 | 0.070 | -0.109 | 0.032 | -0.057 | 0.072 | -0.051 | 0.019 | 0.024 | -0.004 | 0.022 | -0.023 | 0.084 | -0.005 | 1.000 | 0.042 | 0.073 | 0.003 |
| BANK | 0.141 | 0.033 | 0.011 | 0.016 | 0.068 | 0.025 | 0.047 | 0.009 | 0.055 | -0.013 | 0.040 | 0.022 | 0.003 | 0.025 | 0.060 | 0.071 | -0.025 | 0.043 | 0.072 | 0.036 | 0.042 | 1.000 | 0.014 | 0.011 |
| GWEI | 0.011 | 0.030 | 0.041 | -0.087 | -0.066 | -0.030 | 0.068 | -0.029 | 0.008 | -0.037 | 0.033 | -0.018 | 0.016 | 0.101 | 0.050 | 0.064 | 0.044 | 0.110 | 0.016 | -0.023 | 0.073 | 0.014 | 1.000 | 0.049 |
| HMSTR | 0.107 | 0.032 | 0.034 | 0.045 | 0.069 | -0.003 | 0.041 | 0.073 | -0.027 | -0.013 | 0.117 | 0.059 | 0.037 | 0.002 | 0.019 | -0.025 | 0.067 | 0.028 | 0.090 | 0.024 | 0.003 | 0.011 | 0.049 | 1.000 |

The complete 182 × 182 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| ASR | ASRUSDT | 59.1% | 64.0% | ATMUSDT | 0.304 |
| IQ | IQUSDT | 59.5% | 63.6% | BTCUSDT | 0.420 |
| PSG | PSGUSDT | 59.7% | 63.5% | ATMUSDT | 0.348 |
| ONE | ONEUSDT | 60.2% | 63.1% | BTCUSDT | 0.425 |
| ONG | ONGUSDT | 60.4% | 62.9% | BTCUSDT | 0.359 |
| SUN | SUNUSDT | 60.6% | 62.8% | BTCUSDT | 0.312 |
| TRX | TRXUSDT | 60.6% | 62.8% | BTCUSDT | 0.359 |
| TOSHI | TOSHIUSDT | 60.8% | 62.6% | BTCUSDT | 0.536 |
| SPELL | SPELLUSDT | 60.8% | 62.6% | TACUSDT | -0.418 |
| ORCA | ORCAUSDT | 61.0% | 62.5% | BTCUSDT | 0.486 |
| TRUST | TRUSTUSDT | 61.0% | 62.5% | BTCUSDT | 0.480 |
| HOLO | HOLOUSDT | 61.3% | 62.2% | BTCUSDT | 0.367 |
| AI | AIUSDT | 61.5% | 62.1% | BTCUSDT | 0.246 |
| TUT | TUTUSDT | 61.5% | 62.0% | BTCUSDT | 0.376 |
| RAVE | RAVEUSDT | 61.7% | 61.9% | ACTUSDT | 0.369 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

