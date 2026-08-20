# Binance portfolio basis

Generated 2026-07-23T19:09:13.910Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-19T06:00Z through 2026-07-22T23:45Z
- Sampling: exactly 360 15m log returns (3.8 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 1.0% of the maximum unexplained variance, select the largest mean absolute 15m return
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 696 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 12.07 bp | 100.0% | 0 | 1.2B | 31.4% |
| 2 | BANK | BANKUSDT (spot) | spot, usdm-futures | 99.7% | 0.073 | BTCUSDT | 277.82 bp | 100.0% | 0.3 | 97.1M | 849.6% |
| 3 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 99.7% | 0.072 | BTCUSDT | 211.02 bp | 100.0% | 0.3 | 124.1M | 644.8% |
| 4 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 99.7% | 0.061 | ESPORTSUSDT | 203.43 bp | 100.0% | 0.3 | 24.7M | 739.2% |
| 5 | ACE | ACEUSDT (spot) | spot, usdm-futures | 99.8% | 0.044 | BANKUSDT | 155.73 bp | 100.0% | 0.5 | 6.8M | 449.4% |
| 6 | XNO | XNOUSDT (spot) | spot | 99.5% | 0.068 | BTCUSDT | 134.86 bp | 100.0% | 0.8 | 1.2M | 524.6% |
| 7 | LAB | LABUSDT (usdm-futures) | usdm-futures | 99.7% | 0.053 | ESPORTSUSDT | 116.71 bp | 100.0% | 0.5 | 79M | 357.6% |
| 8 | DGB | DGBUSDT (spot) | spot | 99.7% | 0.072 | LABUSDT | 92.45 bp | 100.0% | 1 | 503.6K | 280.2% |
| 9 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 99.4% | 0.075 | LABUSDT | 70.60 bp | 100.0% | 1 | 2.9M | 225.9% |
| 10 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 99.4% | 0.065 | LABUSDT | 54.05 bp | 100.0% | 0.3 | 556.6K | 241.4% |
| 11 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 99.3% | 0.071 | BTCUSDT | 59.28 bp | 100.0% | 0.3 | 5.5M | 148.6% |
| 12 | BR | BRUSDT (usdm-futures) | usdm-futures | 99.3% | 0.065 | DEXEUSDT | 44.94 bp | 100.0% | 0 | 1.4M | 134.9% |
| 13 | M | MUSDT (usdm-futures) | usdm-futures | 99.5% | 0.055 | LABUSDT | 28.94 bp | 100.0% | 0.3 | 1.9M | 74.6% |
| 14 | PIVX | PIVXUSDT (spot) | spot | 99.2% | 0.102 | BTCUSDT | 48.79 bp | 100.0% | 1.3 | 250.8K | 203.7% |
| 15 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 99.1% | 0.071 | DEXEUSDT | 68.09 bp | 100.0% | 0.3 | 8.5M | 188.9% |
| 16 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 99.1% | 0.092 | BTCUSDT | 53.73 bp | 100.0% | 0.3 | 970K | 205.7% |
| 17 | NOW | NOWUSDT (usdm-futures) | usdm-futures | 99.5% | 0.049 | ALLOUSDT | 21.51 bp | 100.0% | 0.5 | 934.8K | 75.1% |
| 18 | PROM | PROMUSDT (spot) | spot, usdm-futures | 98.6% | 0.094 | BANKUSDT | 138.72 bp | 100.0% | 0.5 | 3.8M | 395.7% |
| 19 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 98.7% | 0.067 | NOWUSDT | 36.66 bp | 100.0% | 0.5 | 648.1K | 127.0% |
| 20 | WEN | WENUSDT (usdm-futures) | usdm-futures | 98.7% | 0.115 | ALLOUSDT | 16.77 bp | 100.0% | 1.5 | 474.4K | 45.8% |
| 21 | QI | QIUSDT (spot) | spot | 98.4% | 0.094 | PROMUSDT | 36.52 bp | 100.0% | 3.5 | 124.9K | 103.4% |
| 22 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 98.1% | 0.109 | DGBUSDT | 51.24 bp | 100.0% | 0.3 | 2M | 156.2% |
| 23 | WET | WETUSDT (usdm-futures) | usdm-futures | 98.1% | 0.110 | NOWUSDT | 30.92 bp | 100.0% | 0.3 | 1.3M | 83.2% |
| 24 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 98.4% | 0.104 | BANKUSDT | 20.68 bp | 100.0% | 0.5 | 1.5M | 60.6% |
| 25 | YB | YBUSDT (spot) | spot, usdm-futures | 97.9% | 0.089 | XNOUSDT | 66.29 bp | 100.0% | 0.8 | 1.3M | 182.0% |
| 26 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 97.5% | 0.081 | ACEUSDT | 60.15 bp | 100.0% | 0.3 | 10.1M | 174.5% |
| 27 | MINIMAX | MINIMAXUSDT (usdm-futures) | usdm-futures | 97.6% | 0.095 | BANKUSDT | 56.25 bp | 100.0% | 0.5 | 5.2M | 190.0% |
| 28 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 97.3% | 0.121 | BTCUSDT | 51.53 bp | 100.0% | 0.3 | 2.9M | 140.9% |
| 29 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 97.6% | 0.074 | DGBUSDT | 42.47 bp | 100.0% | 0.5 | 566.2K | 113.9% |
| 30 | ACM | ACMUSDT (spot) | spot | 97.4% | 0.122 | BTCUSDT | 24.35 bp | 100.0% | 1.8 | 210.3K | 62.9% |
| 31 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 97.8% | 0.093 | NOWUSDT | 14.51 bp | 100.0% | 0.8 | 1.1M | 61.2% |
| 32 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 96.7% | 0.101 | WENUSDT | 118.70 bp | 100.0% | 0.3 | 14.3M | 309.4% |
| 33 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 96.6% | 0.135 | GUAUSDT | 51.03 bp | 100.0% | 0.3 | 2.6M | 150.4% |
| 34 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 96.4% | 0.106 | BTCUSDT | 44.23 bp | 100.0% | 0.3 | 1.8M | 115.7% |
| 35 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 96.6% | 0.085 | VELVETUSDT | 33.89 bp | 100.0% | 1.3 | 2M | 98.8% |
| 36 | OPENAI | OPENAIUSDT (usdm-futures) | usdm-futures | 96.4% | 0.116 | LABUSDT | 32.24 bp | 100.0% | 0 | 4.8M | 96.5% |
| 37 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 96.9% | 0.157 | STABLEUSDT | 7.42 bp | 100.0% | 1 | 910.6K | 42.6% |
| 38 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 95.8% | 0.127 | GUAUSDT | 70.75 bp | 100.0% | 0.3 | 10.8M | 171.3% |
| 39 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 95.5% | 0.116 | ACEUSDT | 133.47 bp | 100.0% | 0.3 | 27.3M | 420.0% |
| 40 | LA | LAUSDT (spot) | spot, usdm-futures | 95.7% | 0.105 | BTCUSDT | 69.86 bp | 100.0% | 0.8 | 1.8M | 244.1% |
| 41 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 95.5% | 0.106 | ESPORTSUSDT | 47.43 bp | 100.0% | 0.5 | 2.2M | 128.1% |
| 42 | SKL | SKLUSDT (spot) | spot, usdm-futures | 95.6% | 0.156 | BTCUSDT | 44.12 bp | 100.0% | 1.3 | 908.4K | 160.3% |
| 43 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 95.1% | 0.145 | BRUSDT | 42.56 bp | 100.0% | 0.3 | 4.3M | 127.6% |
| 44 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 95.0% | 0.127 | BROCCOLIF3BUSDT | 42.48 bp | 100.0% | 0.5 | 4.4M | 111.1% |
| 45 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 94.7% | 0.119 | TRUTHUSDT | 87.37 bp | 100.0% | 0.5 | 10.2M | 269.6% |
| 46 | RARE | RAREUSDT (spot) | spot, usdm-futures | 94.6% | 0.157 | BTCUSDT | 25.37 bp | 100.0% | 2.8 | 176K | 82.7% |
| 47 | BABA | BABABUSDT (spot) | spot, usdm-futures | 94.7% | 0.100 | KGENUSDT | 22.73 bp | 100.0% | 0.8 | 553.4K | 67.4% |
| 48 | KAVA | KAVAUSDT (spot) | spot, usdm-futures | 94.8% | 0.168 | BTCUSDT | 15.32 bp | 100.0% | 0.8 | 322.4K | 62.0% |
| 49 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 94.7% | 0.114 | MUSDT | 11.84 bp | 100.0% | 1 | 18.6M | 31.9% |
| 50 | U | UUSDT (spot) | spot | 94.8% | 0.104 | GENIUSUSDT | 0.47 bp | 100.0% | 3.8 | 15.3M | 1.3% |
| 51 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 93.9% | 0.121 | GENIUSUSDT | 54.91 bp | 100.0% | 0.5 | 3.9M | 151.4% |
| 52 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 93.6% | 0.105 | GENIUSUSDT | 56.99 bp | 100.0% | 0.3 | 5.1M | 173.0% |
| 53 | TAC | TACUSDT (usdm-futures) | usdm-futures | 93.4% | 0.152 | MUSDT | 73.51 bp | 100.0% | 0.3 | 4.6M | 184.9% |
| 54 | TUT | TUTUSDT (spot) | spot, usdm-futures | 93.6% | 0.114 | PTBUSDT | 51.41 bp | 100.0% | 0.3 | 887.2K | 131.4% |
| 55 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 93.4% | 0.151 | BTCUSDT | 34.78 bp | 100.0% | 0.5 | 1.2M | 100.3% |
| 56 | TREE | TREEUSDT (spot) | spot, usdm-futures | 93.1% | 0.137 | BTCUSDT | 58.47 bp | 100.0% | 1 | 2.2M | 238.4% |
| 57 | REQ | REQUSDT (spot) | spot | 93.4% | 0.232 | BTCUSDT | 26.91 bp | 100.0% | 1.5 | 220.3K | 127.6% |
| 58 | MMT | MMTUSDT (spot) | spot, usdm-futures | 92.7% | 0.119 | SKLUSDT | 35.47 bp | 100.0% | 0.5 | 667.7K | 90.2% |
| 59 | GEV | GEVUSDT (usdm-futures) | usdm-futures | 92.7% | 0.207 | EPICUSDT | 26.83 bp | 100.0% | 0.5 | 443.4K | 107.9% |
| 60 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 92.4% | 0.161 | BTCUSDT | 50.58 bp | 100.0% | 0.5 | 6.8M | 139.8% |
| 61 | OPN | OPNUSDT (spot) | spot, usdm-futures | 92.1% | 0.115 | BTCUSDT | 43.52 bp | 100.0% | 1.3 | 32.3M | 122.3% |
| 62 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 91.8% | 0.114 | VELVETUSDT | 65.45 bp | 100.0% | 0.5 | 5.8M | 187.3% |
| 63 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 92.0% | 0.155 | TREEUSDT | 55.36 bp | 100.0% | 0.3 | 2.1M | 147.8% |
| 64 | MET | METUSDT (spot) | spot, usdm-futures | 91.6% | 0.130 | NOWUSDT | 53.09 bp | 100.0% | 0.3 | 1.5M | 137.3% |
| 65 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 91.4% | 0.174 | BRKBUSDT | 29.71 bp | 100.0% | 0.8 | 1.9M | 82.0% |
| 66 | Q | QUSDT (usdm-futures) | usdm-futures | 91.5% | 0.113 | RECALLUSDT | 25.93 bp | 100.0% | 0.3 | 848K | 82.4% |
| 67 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 91.1% | 0.150 | PROMUSDT | 52.64 bp | 100.0% | 0.3 | 3.9M | 164.1% |
| 68 | AI | AIUSDT (spot) | spot | 91.5% | 0.140 | BANKUSDT | 24.38 bp | 100.0% | 3.5 | 234.2K | 71.1% |
| 69 | TURTLE | TURTLEUSDT (spot) | spot, usdm-futures | 90.9% | 0.221 | LABUSDT | 38.66 bp | 100.0% | 1.8 | 508K | 138.4% |
| 70 | AMP | AMPUSDT (spot) | spot | 91.2% | 0.146 | WENUSDT | 23.47 bp | 100.0% | 1.5 | 383.3K | 70.6% |
| 71 | ESP | ESPUSDT (spot) | spot, usdm-futures | 91.3% | 0.113 | ZESTUSDT | 20.22 bp | 100.0% | 0.3 | 193.8K | 51.5% |
| 72 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 90.3% | 0.204 | BANKUSDT | 49.92 bp | 100.0% | 0.3 | 2.1M | 140.0% |
| 73 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 90.0% | 0.110 | BTCUSDT | 53.45 bp | 100.0% | 2.3 | 88K | 160.2% |
| 74 | ZIL | ZILUSDT (spot) | spot, usdm-futures | 90.2% | 0.178 | BTCUSDT | 38.85 bp | 100.0% | 1.3 | 1.2M | 120.8% |
| 75 | BTTC | BTTCUSDT (spot) | spot | 89.7% | 0.128 | LABUSDT | 93.95 bp | 100.0% | 8.5 | 124.6K | 346.5% |
| 76 | LISTA | LISTAUSDT (spot) | spot, usdm-futures | 90.1% | 0.203 | PIVXUSDT | 38.15 bp | 100.0% | 2.3 | 212.8K | 199.2% |
| 77 | STRC | STRCUSDT (usdm-futures) | usdm-futures | 90.0% | 0.202 | BTCUSDT | 10.09 bp | 100.0% | 0.3 | 1.4M | 30.6% |
| 78 | ATM | ATMUSDT (spot) | spot | 89.5% | 0.180 | DEXEUSDT | 63.13 bp | 100.0% | 0.5 | 1.1M | 332.4% |
| 79 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 89.2% | 0.173 | DEXEUSDT | 54.85 bp | 100.0% | 0.3 | 3.6M | 188.5% |
| 80 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 89.2% | 0.132 | TRUTHUSDT | 37.65 bp | 100.0% | 0.3 | 962.6K | 93.7% |
| 81 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 88.6% | 0.133 | BTCUSDT | 46.83 bp | 100.0% | 1.5 | 1.6M | 133.9% |
| 82 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 88.4% | 0.139 | COOKIEUSDT | 28.99 bp | 100.0% | 0.5 | 3M | 78.6% |
| 83 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 88.8% | 0.159 | TAGUSDT | 23.39 bp | 100.0% | 1 | 255.3K | 77.5% |
| 84 | O | OUSDT (usdm-futures) | usdm-futures | 88.1% | 0.134 | PHAROSUSDT | 44.88 bp | 100.0% | 0.3 | 4.8M | 138.8% |
| 85 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 88.2% | 0.153 | BTCUSDT | 32.55 bp | 100.0% | 0.3 | 4.3M | 110.6% |
| 86 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 87.8% | 0.185 | WETUSDT | 35.08 bp | 100.0% | 0.8 | 2.4M | 110.7% |
| 87 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 87.7% | 0.140 | MANTRAUSDT | 53.41 bp | 100.0% | 0.8 | 1.6M | 139.3% |
| 88 | BEL | BELUSDT (spot) | spot, usdm-futures | 87.8% | 0.170 | BTCUSDT | 32.13 bp | 100.0% | 0.8 | 427K | 81.7% |
| 89 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 87.5% | 0.216 | VELVETUSDT | 74.96 bp | 100.0% | 0.3 | 2.6M | 303.1% |
| 90 | BX | BXUSDT (usdm-futures) | usdm-futures | 87.8% | 0.176 | BTCUSDT | 15.84 bp | 100.0% | 0.3 | 869.3K | 43.8% |
| 91 | PYR | PYRUSDT (spot) | spot | 86.7% | 0.170 | BROCCOLIF3BUSDT | 83.87 bp | 100.0% | 1 | 1M | 216.0% |
| 92 | PORTO | PORTOUSDT (spot) | spot | 86.6% | 0.164 | DEXEUSDT | 63.49 bp | 100.0% | 0.8 | 544.6K | 258.7% |
| 93 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 86.5% | 0.165 | MINIMAXUSDT | 30.62 bp | 100.0% | 0.8 | 1.5M | 80.5% |
| 94 | BAS | BASUSDT (usdm-futures) | usdm-futures | 86.3% | 0.144 | AMPUSDT | 42.16 bp | 100.0% | 0.3 | 2.3M | 110.1% |
| 95 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 86.0% | 0.252 | ACEUSDT | 20.39 bp | 100.0% | 1.8 | 179.8K | 82.8% |
| 96 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 86.3% | 0.162 | BTCUSDT | 10.32 bp | 100.0% | 1.3 | 177.5K | 31.7% |
| 97 | SUN | SUNUSDT (spot) | spot, usdm-futures | 86.3% | 0.236 | BTCUSDT | 8.30 bp | 100.0% | 1.5 | 338.7K | 23.7% |
| 98 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 85.1% | 0.141 | STRCUSDT | 161.80 bp | 100.0% | 0 | 215.7M | 431.8% |
| 99 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 85.1% | 0.135 | BTCUSDT | 58.32 bp | 100.0% | 0.3 | 4.7M | 149.1% |
| 100 | AIN | AINUSDT (usdm-futures) | usdm-futures | 85.3% | 0.156 | LIGHTUSDT | 52.05 bp | 100.0% | 0.3 | 1.2M | 151.4% |
| 101 | POLYX | POLYXUSDT (spot) | spot, usdm-futures | 84.7% | 0.193 | BTCUSDT | 30.43 bp | 100.0% | 1.5 | 360.6K | 87.4% |
| 102 | TA | TAUSDT (usdm-futures) | usdm-futures | 84.9% | 0.130 | PYRUSDT | 29.86 bp | 100.0% | 0.5 | 1.7M | 84.1% |
| 103 | IBM | IBMBUSDT (spot) | spot, usdm-futures | 85.0% | 0.207 | GOOGLBUSDT | 21.35 bp | 100.0% | 0.5 | 814.3K | 74.1% |
| 104 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 84.0% | 0.130 | PYRUSDT | 73.08 bp | 100.0% | 0.8 | 6M | 189.3% |
| 105 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 83.8% | 0.177 | BLUAIUSDT | 75.39 bp | 100.0% | 1.3 | 6.6M | 212.1% |
| 106 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 83.7% | 0.178 | BTCUSDT | 42.63 bp | 100.0% | 0.3 | 1.2M | 111.7% |
| 107 | VANA | VANAUSDT (spot) | spot, usdm-futures | 84.0% | 0.174 | BTCUSDT | 30.42 bp | 100.0% | 0.8 | 982.3K | 82.1% |
| 108 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 83.6% | 0.149 | FOLKSUSDT | 41.51 bp | 100.0% | 0.3 | 1.7M | 106.1% |
| 109 | SOLV | SOLVUSDT (spot) | spot, usdm-futures | 82.9% | 0.245 | ESPORTSUSDT | 55.35 bp | 100.0% | 3.5 | 1M | 192.4% |
| 110 | TST | TSTUSDT (spot) | spot, usdm-futures | 82.6% | 0.161 | KAITOUSDT | 38.12 bp | 100.0% | 0.5 | 320.3K | 103.4% |
| 111 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 82.5% | 0.187 | LABUSDT | 52.53 bp | 100.0% | 0.3 | 4.9M | 137.4% |
| 112 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 82.5% | 0.301 | BTCUSDT | 31.58 bp | 100.0% | 0.5 | 325.4K | 86.2% |
| 113 | NOK | NOKBUSDT (spot) | spot, usdm-futures | 82.6% | 0.224 | IBMBUSDT | 31.06 bp | 100.0% | 0.5 | 163.8K | 81.6% |
| 114 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 82.7% | 0.157 | BTCUSDT | 17.87 bp | 100.0% | 0.3 | 480.8K | 51.7% |
| 115 | GTC | GTCUSDT (spot) | spot, usdm-futures | 81.9% | 0.149 | BILLUSDT | 51.17 bp | 100.0% | 2.8 | 151.5K | 158.8% |
| 116 | SXT | SXTUSDT (spot) | spot, usdm-futures | 82.1% | 0.183 | BASUSDT | 36.50 bp | 100.0% | 0.5 | 790.8K | 97.0% |
| 117 | CC | CCUSDT (usdm-futures) | usdm-futures | 82.3% | 0.170 | BTCUSDT | 17.00 bp | 100.0% | 0.3 | 2.6M | 44.7% |
| 118 | TLM | TLMUSDT (spot) | spot, usdm-futures | 81.2% | 0.138 | UUSDT | 135.17 bp | 100.0% | 0.3 | 7.7M | 395.4% |
| 119 | DODOX | DODOXUSDT (usdm-futures) | usdm-futures | 81.0% | 0.140 | GENIUSUSDT | 84.40 bp | 100.0% | 0 | 11.5M | 240.7% |
| 120 | RPL | RPLUSDT (spot) | spot, usdm-futures | 81.0% | 0.187 | ACMUSDT | 40.71 bp | 100.0% | 1.3 | 226.1K | 107.1% |
| 121 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 80.6% | 0.173 | LYNUSDT | 38.40 bp | 100.0% | 0.5 | 1.1M | 103.5% |
| 122 | AWE | AWEUSDT (spot) | spot, usdm-futures | 80.6% | 0.148 | LIGHTUSDT | 27.19 bp | 100.0% | 0.8 | 190.5K | 92.4% |
| 123 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 80.4% | 0.162 | BTCUSDT | 54.40 bp | 100.0% | 0.3 | 7M | 145.2% |
| 124 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 80.7% | 0.322 | GOOGLBUSDT | 9.09 bp | 100.0% | 0.3 | 8.3M | 31.1% |
| 125 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 80.1% | 0.148 | KGENUSDT | 57.90 bp | 100.0% | 0.3 | 4.1M | 150.6% |
| 126 | MITO | MITOUSDT (spot) | spot, usdm-futures | 79.7% | 0.165 | AMPUSDT | 51.04 bp | 100.0% | 0.3 | 1.1M | 141.4% |
| 127 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 79.9% | 0.180 | AIOTUSDT | 46.46 bp | 100.0% | 0.3 | 1.7M | 124.4% |
| 128 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 80.0% | 0.175 | AINUSDT | 7.26 bp | 100.0% | 0.8 | 346.1K | 20.7% |
| 129 | JST | JSTUSDT (spot) | spot, usdm-futures | 79.5% | 0.212 | GEVUSDT | 13.41 bp | 100.0% | 0.5 | 2.3M | 35.6% |
| 130 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 78.9% | 0.138 | BASUSDT | 58.81 bp | 100.0% | 1 | 5.4M | 153.1% |
| 131 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 78.7% | 0.169 | KAITOUSDT | 77.47 bp | 100.0% | 0.3 | 4.4M | 246.6% |
| 132 | HEI | HEIUSDT (spot) | spot, usdm-futures | 78.6% | 0.178 | PTBUSDT | 54.42 bp | 100.0% | 0.5 | 1.6M | 142.4% |
| 133 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 78.7% | 0.146 | SOLVUSDT | 38.73 bp | 100.0% | 0.3 | 10.2M | 100.5% |
| 134 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 78.3% | 0.231 | TREEUSDT | 38.97 bp | 100.0% | 0.8 | 791.1K | 98.8% |
| 135 | THE | THEUSDT (spot) | spot, usdm-futures | 78.1% | 0.219 | BTCUSDT | 36.69 bp | 100.0% | 1 | 782.4K | 115.4% |
| 136 | ZORA | ZORAUSDT (usdm-futures) | usdm-futures | 78.4% | 0.234 | BTCUSDT | 31.95 bp | 100.0% | 0.3 | 2.2M | 89.1% |
| 137 | XAN | XANUSDT (usdm-futures) | usdm-futures | 77.8% | 0.162 | REQUSDT | 67.65 bp | 100.0% | 0.3 | 5.3M | 195.1% |
| 138 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 77.8% | 0.361 | OPENAIUSDT | 26.95 bp | 100.0% | 0.3 | 9.2M | 118.3% |
| 139 | IWM | IWMUSDT (usdm-futures) | usdm-futures | 78.0% | 0.212 | SPACEUSDT | 8.74 bp | 100.0% | 1.3 | 378.4K | 36.1% |
| 140 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 77.5% | 0.170 | TAKEUSDT | 35.01 bp | 100.0% | 0.5 | 2.9M | 93.1% |
| 141 | US | USUSDT (usdm-futures) | usdm-futures | 76.6% | 0.176 | TAGUSDT | 96.39 bp | 100.0% | 0 | 26.5M | 289.6% |
| 142 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 76.7% | 0.163 | BTCUSDT | 16.19 bp | 100.0% | 1.5 | 2.9M | 47.7% |
| 143 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 76.7% | 0.169 | STRCUSDT | 9.19 bp | 100.0% | 1.3 | 163K | 28.7% |
| 144 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 76.2% | 0.186 | PORTOUSDT | 61.55 bp | 100.0% | 0.5 | 4.3M | 161.7% |
| 145 | DOOD | DOODUSDT (usdm-futures) | usdm-futures | 75.7% | 0.207 | BTCUSDT | 33.80 bp | 100.0% | 0.5 | 2.1M | 92.4% |
| 146 | BIO | BIOUSDT (spot) | spot, usdm-futures | 75.6% | 0.326 | BTCUSDT | 31.66 bp | 100.0% | 0.5 | 1.4M | 89.1% |
| 147 | WIN | WINUSDT (spot) | spot | 75.4% | 0.261 | BTCUSDT | 25.20 bp | 100.0% | 0.5 | 119K | 66.2% |
| 148 | APR | APRUSDT (usdm-futures) | usdm-futures | 75.6% | 0.170 | LISTAUSDT | 24.80 bp | 100.0% | 0.5 | 1.7M | 64.5% |
| 149 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 75.7% | 0.185 | BXUSDT | 11.36 bp | 100.0% | 0.5 | 2M | 35.9% |
| 150 | ON | ONUSDT (usdm-futures) | usdm-futures | 74.7% | 0.175 | ANKRUSDT | 158.92 bp | 100.0% | 0 | 25.3M | 436.8% |
| 151 | ERA | ERAUSDT (spot) | spot, usdm-futures | 74.6% | 0.269 | LAUSDT | 133.58 bp | 100.0% | 0.8 | 10.4M | 514.5% |
| 152 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 74.4% | 0.292 | BROCCOLIF3BUSDT | 98.05 bp | 100.0% | 0.3 | 8.3M | 355.2% |
| 153 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 74.2% | 0.161 | SPACEUSDT | 38.42 bp | 100.0% | 0.5 | 897.8K | 105.5% |
| 154 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 74.1% | 0.265 | BTCUSDT | 23.52 bp | 100.0% | 0.3 | 707.1K | 62.9% |
| 155 | SAGA | SAGAUSDT (spot) | spot, usdm-futures | 73.9% | 0.225 | ANKRUSDT | 41.29 bp | 100.0% | 0.5 | 1.2M | 121.7% |
| 156 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 73.7% | 0.251 | BTCUSDT | 26.38 bp | 100.0% | 0.3 | 284K | 65.9% |
| 157 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 73.9% | 0.194 | BTCUSDT | 22.07 bp | 100.0% | 0.3 | 1.9M | 75.3% |
| 158 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 73.3% | 0.322 | BTCUSDT | 27.18 bp | 100.0% | 1 | 725.5K | 72.4% |
| 159 | RAD | RADUSDT (spot) | spot | 73.5% | 0.226 | AIUSDT | 24.87 bp | 100.0% | 5 | 131.5K | 66.1% |
| 160 | XEC | XECUSDT (spot) | spot, usdm-futures | 72.9% | 0.181 | LABUSDT | 68.77 bp | 100.0% | 0.5 | 2.7M | 185.5% |
| 161 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 73.1% | 0.289 | BTCUSDT | 14.14 bp | 100.0% | 0.8 | 1.1M | 37.3% |
| 162 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 72.6% | 0.175 | BTCUSDT | 73.02 bp | 100.0% | 0.3 | 10.3M | 216.8% |
| 163 | PLUME | PLUMEUSDT (spot) | spot, usdm-futures | 72.4% | 0.271 | BTCUSDT | 44.05 bp | 100.0% | 0.5 | 1.1M | 109.1% |
| 164 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 72.7% | 0.368 | BTCUSDT | 19.81 bp | 100.0% | 1 | 116.6K | 55.4% |
| 165 | BAR | BARUSDT (spot) | spot | 72.0% | 0.190 | BABABUSDT | 30.24 bp | 100.0% | 2 | 564.3K | 85.7% |
| 166 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 71.3% | 0.246 | BTCUSDT | 36.14 bp | 100.0% | 0.3 | 2.6M | 101.9% |
| 167 | BAN | BANUSDT (usdm-futures) | usdm-futures | 71.4% | 0.176 | AMPUSDT | 22.81 bp | 100.0% | 0.5 | 1.3M | 60.8% |
| 168 | CTSI | CTSIUSDT (spot) | spot, usdm-futures | 71.4% | 0.285 | BELUSDT | 22.80 bp | 100.0% | 0.8 | 251.2K | 82.2% |
| 169 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 71.3% | 0.309 | BTCUSDT | 17.62 bp | 100.0% | 0.8 | 118K | 46.8% |
| 170 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 70.7% | 0.195 | LAUSDT | 30.44 bp | 100.0% | 0.3 | 798.5K | 83.1% |
| 171 | XVS | XVSUSDT (spot) | spot, usdm-futures | 70.6% | 0.380 | BTCUSDT | 18.54 bp | 100.0% | 2.5 | 78.1K | 56.7% |
| 172 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 70.3% | 0.176 | PORTOUSDT | 59.55 bp | 100.0% | 0.3 | 2.8M | 167.7% |
| 173 | OPG | OPGUSDT (spot) | spot, usdm-futures | 70.2% | 0.211 | BTCUSDT | 30.57 bp | 100.0% | 0.5 | 764.9K | 82.6% |
| 174 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 70.2% | 0.230 | BTCUSDT | 28.72 bp | 100.0% | 0.5 | 1.1M | 72.2% |
| 175 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 70.3% | 0.156 | BLUAIUSDT | 11.15 bp | 100.0% | 0.8 | 320K | 33.6% |
| 176 | STAR | STARUSDT (usdm-futures) | usdm-futures | 69.8% | 0.170 | TREEUSDT | 62.70 bp | 100.0% | 0.3 | 2.3M | 163.3% |
| 177 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 69.3% | 0.265 | SKLUSDT | 58.40 bp | 100.0% | 0.3 | 2.5M | 152.0% |
| 178 | TKO | TKOUSDT (spot) | spot | 69.1% | 0.223 | OUSDT | 30.88 bp | 100.0% | 1.3 | 150.6K | 97.3% |
| 179 | MTL | MTLUSDT (spot) | spot, usdm-futures | 69.2% | 0.276 | BTCUSDT | 14.26 bp | 100.0% | 3.5 | 24.9K | 52.4% |
| 180 | CGPT | CGPTUSDT (spot) | spot, usdm-futures | 68.5% | 0.316 | BTCUSDT | 31.06 bp | 100.0% | 0.3 | 309.8K | 80.1% |
| 181 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 68.2% | 0.292 | LISTAUSDT | 32.82 bp | 100.0% | 0.5 | 534.3K | 85.9% |
| 182 | WAL | WALUSDT (spot) | spot, usdm-futures | 68.4% | 0.365 | BTCUSDT | 28.98 bp | 100.0% | 1.3 | 298K | 75.6% |
| 183 | MBL | MBLUSDT (spot) | spot | 68.1% | 0.176 | BTCUSDT | 21.15 bp | 100.0% | 1 | 346.1K | 55.5% |
| 184 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 67.4% | 0.229 | VELVETUSDT | 59.00 bp | 100.0% | 0.3 | 17.4M | 156.7% |
| 185 | ACT | ACTUSDT (spot) | spot, usdm-futures | 67.7% | 0.263 | KAVAUSDT | 40.81 bp | 100.0% | 0.5 | 669.2K | 116.9% |
| 186 | NOM | NOMUSDT (spot) | spot, usdm-futures | 67.5% | 0.183 | BTCUSDT | 37.43 bp | 100.0% | 2.8 | 528.1K | 110.5% |
| 187 | C | CUSDT (spot) | spot, usdm-futures | 67.6% | 0.283 | BTCUSDT | 32.18 bp | 100.0% | 1 | 363.3K | 91.3% |
| 188 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 66.9% | 0.350 | AMPUSDT | 26.27 bp | 100.0% | 0.8 | 642.4K | 78.9% |
| 189 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 66.9% | 0.311 | BTCUSDT | 23.26 bp | 100.0% | 0.5 | 826.2K | 60.1% |
| 190 | AZTEC | AZTECUSDT (usdm-futures) | usdm-futures | 66.5% | 0.351 | BTCUSDT | 32.76 bp | 100.0% | 0.5 | 2.5M | 112.4% |
| 191 | ICX | ICXUSDT (spot) | spot, usdm-futures | 66.7% | 0.367 | BTCUSDT | 19.32 bp | 100.0% | 2.3 | 41.1K | 61.5% |
| 192 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 66.0% | 0.280 | BTCUSDT | 13.10 bp | 100.0% | 5 | 292.7K | 38.5% |
| 193 | UB | UBUSDT (usdm-futures) | usdm-futures | 65.7% | 0.174 | VANAUSDT | 74.61 bp | 100.0% | 0.3 | 21.2M | 191.3% |
| 194 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 65.8% | 0.201 | GUAUSDT | 42.21 bp | 100.0% | 0.3 | 3M | 111.0% |
| 195 | ADX | ADXUSDT (spot) | spot | 65.6% | 0.249 | BTCUSDT | 19.51 bp | 100.0% | 2.3 | 237.1K | 49.8% |
| 196 | BONK | BONKUSDT (spot) | spot, usdm-futures | 64.8% | 0.310 | BTCUSDT | 53.34 bp | 100.0% | 1 | 3.7M | 141.4% |
| 197 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 64.8% | 0.239 | BTCUSDT | 46.98 bp | 100.0% | 0.3 | 9.7M | 119.4% |
| 198 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 64.8% | 0.249 | BTCUSDT | 32.71 bp | 100.0% | 0.3 | 1.2M | 91.2% |
| 199 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 64.6% | 0.263 | NFLXUSDT | 24.48 bp | 100.0% | 0.3 | 17.1M | 72.6% |
| 200 | SPK | SPKUSDT (spot) | spot, usdm-futures | 64.2% | 0.226 | BTCUSDT | 18.09 bp | 100.0% | 0.3 | 571.5K | 48.9% |
| 201 | JUV | JUVUSDT (spot) | spot | 63.8% | 0.153 | AIUSDT | 22.07 bp | 100.0% | 2 | 252.1K | 62.1% |
| 202 | PUMP | PUMPUSDT (spot) | spot, usdm-futures | 63.6% | 0.251 | BTCUSDT | 48.30 bp | 100.0% | 0.5 | 10.4M | 128.8% |
| 203 | V | VUSDT (usdm-futures) | usdm-futures | 63.8% | 0.235 | BXUSDT | 7.57 bp | 100.0% | 0.5 | 265.8K | 21.7% |
| 204 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 63.3% | 0.240 | BIOUSDT | 45.74 bp | 100.0% | 0.3 | 5.8M | 127.3% |
| 205 | SNX | SNXUSDT (spot) | spot, usdm-futures | 62.7% | 0.242 | AZTECUSDT | 40.34 bp | 100.0% | 2.3 | 1.3M | 113.6% |
| 206 | PHA | PHAUSDT (spot) | spot, usdm-futures | 62.8% | 0.336 | BTCUSDT | 38.94 bp | 100.0% | 1 | 756.4K | 100.2% |
| 207 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 62.3% | 0.246 | BTCUSDT | 34.06 bp | 100.0% | 0.5 | 689.9K | 89.1% |
| 208 | QUICK | QUICKUSDT (spot) | spot | 62.3% | 0.170 | SOLVUSDT | 33.27 bp | 100.0% | 0.8 | 41.2K | 83.0% |
| 209 | XPL | XPLUSDT (spot) | spot, usdm-futures | 62.3% | 0.354 | BTCUSDT | 31.18 bp | 100.0% | 0.3 | 4.6M | 78.9% |
| 210 | IQ | IQUSDT (spot) | spot | 62.3% | 0.209 | CTSIUSDT | 16.25 bp | 100.0% | 1 | 26.2K | 42.6% |
| 211 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 61.8% | 0.219 | KMNOUSDT | 40.57 bp | 100.0% | 1 | 1.1M | 124.3% |
| 212 | GRAM | GRAMUSDT (spot) | spot, usdm-futures | 61.8% | 0.284 | BTCUSDT | 28.47 bp | 100.0% | 0.5 | 8M | 97.8% |
| 213 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 61.2% | 0.224 | KMNOUSDT | 37.83 bp | 100.0% | 0.5 | 2.5M | 99.2% |
| 214 | ACX | ACXUSDT (spot) | spot, usdm-futures | 61.4% | 0.228 | BTCUSDT | 11.05 bp | 100.0% | 1 | 55.1K | 32.4% |
| 215 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 60.8% | 0.264 | LABUSDT | 46.52 bp | 100.0% | 0.3 | 14.5M | 121.8% |
| 216 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 60.7% | 0.226 | KAITOUSDT | 43.75 bp | 100.0% | 0.3 | 7.6M | 108.8% |
| 217 | SCRT | SCRTUSDT (spot) | spot, usdm-futures | 60.2% | 0.252 | OPGUSDT | 32.13 bp | 100.0% | 1 | 217.7K | 88.5% |
| 218 | 1INCH | 1INCHUSDT (spot) | spot, usdm-futures | 60.1% | 0.401 | BTCUSDT | 30.16 bp | 100.0% | 0.8 | 486.2K | 81.4% |
| 219 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 60.3% | 0.186 | SPCXBUSDT | 7.16 bp | 100.0% | 0.5 | 487.2K | 23.7% |
| 220 | CAT | 1000CATUSDT (spot) | spot, usdm-futures | 59.5% | 0.268 | BTCUSDT | 34.35 bp | 100.0% | 2.3 | 79.2K | 102.9% |
| 221 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 59.2% | 0.351 | MINIMAXUSDT | 33.18 bp | 100.0% | 0.5 | 3.8M | 96.7% |
| 222 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 59.3% | 0.285 | WALUSDT | 26.71 bp | 100.0% | 0.5 | 1.7M | 69.3% |
| 223 | RIF | RIFUSDT (spot) | spot, usdm-futures | 58.5% | 0.215 | GEVUSDT | 134.04 bp | 100.0% | 0.5 | 3.7M | 418.0% |
| 224 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 58.3% | 0.273 | BLUAIUSDT | 48.02 bp | 100.0% | 0.3 | 3.8M | 156.8% |
| 225 | VVV | VVVUSDT (usdm-futures) | usdm-futures | 58.3% | 0.409 | BTCUSDT | 42.76 bp | 100.0% | 0.3 | 14.8M | 115.8% |
| 226 | 4 | 4USDT (usdm-futures) | usdm-futures | 57.9% | 0.273 | BTCUSDT | 33.29 bp | 100.0% | 0.3 | 1.3M | 99.2% |
| 227 | BABY | BABYUSDT (spot) | spot, usdm-futures | 57.9% | 0.382 | BTCUSDT | 25.36 bp | 100.0% | 0.5 | 237K | 68.5% |
| 228 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 57.6% | 0.220 | ZORAUSDT | 34.06 bp | 100.0% | 0.3 | 1.4M | 103.1% |
| 229 | XBI | XBIUSDT (usdm-futures) | usdm-futures | 57.9% | 0.229 | BTCUSDT | 10.11 bp | 100.0% | 0.5 | 301.8K | 28.1% |
| 230 | RE | REUSDT (spot) | spot, usdm-futures | 56.6% | 0.204 | VANAUSDT | 66.14 bp | 100.0% | 0.3 | 57.9M | 166.9% |
| 231 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 56.7% | 0.182 | ICXUSDT | 39.24 bp | 100.0% | 0.3 | 1.3M | 115.7% |
| 232 | ZKP | ZKPUSDT (spot) | spot, usdm-futures | 56.5% | 0.294 | BTCUSDT | 27.62 bp | 100.0% | 1.8 | 288.4K | 79.6% |

## Diagnostics

- Basis size selected: 232
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.054
- Maximum pairwise absolute correlation: 0.409
- Mean whole-market projection R²: 84.5%
- Median whole-market projection R²: 80.0%
- 10th-percentile whole-market projection R²: 71.5%
- Minimum whole-market projection R²: 68.3%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.7% | 7.8% | 0.4% | 0.0% |
| 5 | 14.6% | 10.0% | 1.5% | 0.1% |
| 10 | 16.6% | 11.7% | 3.0% | 0.4% |
| 15 | 18.5% | 13.2% | 4.5% | 1.0% |
| 20 | 20.8% | 15.4% | 6.0% | 2.4% |
| 25 | 22.8% | 17.0% | 7.6% | 4.2% |
| 30 | 25.1% | 19.0% | 9.1% | 4.4% |
| 35 | 26.9% | 20.4% | 10.8% | 6.2% |
| 40 | 29.2% | 22.1% | 12.5% | 8.1% |
| 45 | 31.0% | 23.9% | 13.9% | 9.8% |
| 50 | 33.0% | 25.7% | 15.6% | 11.2% |
| 55 | 35.0% | 27.4% | 17.7% | 12.5% |
| 60 | 37.1% | 29.7% | 18.9% | 14.4% |
| 65 | 39.1% | 31.6% | 20.5% | 15.9% |
| 70 | 40.8% | 33.1% | 22.1% | 16.7% |
| 75 | 42.5% | 34.3% | 23.7% | 18.8% |
| 80 | 44.6% | 36.6% | 25.4% | 21.1% |
| 85 | 46.2% | 38.0% | 26.9% | 22.2% |
| 90 | 47.8% | 39.9% | 28.6% | 24.3% |
| 95 | 49.6% | 41.6% | 30.5% | 25.5% |
| 100 | 51.2% | 43.3% | 32.0% | 27.7% |
| 105 | 52.9% | 45.0% | 33.6% | 29.3% |
| 110 | 54.4% | 46.3% | 34.9% | 31.4% |
| 115 | 56.2% | 48.7% | 37.1% | 32.3% |
| 120 | 57.7% | 50.3% | 38.7% | 34.5% |
| 125 | 59.3% | 51.9% | 40.3% | 35.9% |
| 130 | 60.8% | 53.6% | 41.7% | 37.5% |
| 135 | 62.2% | 55.0% | 43.5% | 38.5% |
| 140 | 63.7% | 56.3% | 45.0% | 40.8% |
| 145 | 65.0% | 58.0% | 46.5% | 42.5% |
| 150 | 66.3% | 58.9% | 48.0% | 44.4% |
| 155 | 67.5% | 60.3% | 49.4% | 45.2% |
| 160 | 69.0% | 61.9% | 50.8% | 46.5% |
| 165 | 70.3% | 63.7% | 52.3% | 48.8% |
| 170 | 71.5% | 65.0% | 53.7% | 49.9% |
| 175 | 72.7% | 66.2% | 55.1% | 51.2% |
| 180 | 73.8% | 67.2% | 56.6% | 53.2% |
| 185 | 74.9% | 68.3% | 57.9% | 54.3% |
| 190 | 76.0% | 69.6% | 59.2% | 55.4% |
| 195 | 77.1% | 70.9% | 60.9% | 57.5% |
| 200 | 78.3% | 72.4% | 62.5% | 59.0% |
| 205 | 79.3% | 73.7% | 63.8% | 60.5% |
| 210 | 80.4% | 74.9% | 65.2% | 61.8% |
| 215 | 81.4% | 76.0% | 66.4% | 63.1% |
| 220 | 82.3% | 77.4% | 67.8% | 64.8% |
| 225 | 83.2% | 78.3% | 69.2% | 66.1% |
| 230 | 84.1% | 79.3% | 70.8% | 67.8% |
| 232 | 84.5% | 80.0% | 71.5% | 68.3% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | BANK | ESPORTS | DEXE | ACE | XNO | LAB | DGB | EPIC | BROCCOLIF3B | ALLO | BR | M | PIVX | BTW | NAORIS | NOW | PROM | GENIUS | WEN | QI | GUA | WET | STABLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.073 | 0.072 | -0.008 | 0.038 | -0.068 | -0.009 | 0.015 | 0.006 | 0.051 | 0.071 | 0.039 | 0.028 | -0.102 | 0.048 | 0.092 | -0.009 | -0.006 | 0.062 | 0.058 | -0.013 | -0.034 | 0.087 | 0.078 |
| BANK | -0.073 | 1.000 | -0.006 | 0.050 | -0.044 | -0.038 | -0.030 | 0.026 | 0.029 | -0.046 | 0.041 | 0.005 | 0.004 | -0.014 | -0.056 | 0.014 | 0.017 | 0.094 | 0.044 | -0.048 | 0.026 | 0.032 | -0.040 | -0.104 |
| ESPORTS | 0.072 | -0.006 | 1.000 | -0.061 | 0.030 | -0.040 | 0.053 | -0.005 | 0.037 | -0.031 | 0.032 | -0.006 | 0.021 | -0.001 | 0.038 | -0.036 | 0.024 | 0.029 | 0.059 | 0.006 | 0.003 | 0.032 | -0.014 | 0.039 |
| DEXE | -0.008 | 0.050 | -0.061 | 1.000 | -0.028 | 0.021 | -0.052 | -0.006 | -0.011 | -0.015 | -0.027 | -0.065 | 0.001 | 0.031 | 0.071 | 0.014 | -0.006 | 0.057 | -0.055 | 0.005 | 0.046 | -0.011 | -0.035 | 0.019 |
| ACE | 0.038 | -0.044 | 0.030 | -0.028 | 1.000 | 0.035 | 0.016 | -0.018 | -0.055 | -0.023 | -0.041 | 0.045 | -0.012 | -0.018 | -0.009 | 0.033 | 0.014 | -0.042 | 0.051 | 0.020 | -0.030 | -0.002 | -0.030 | -0.004 |
| XNO | -0.068 | -0.038 | -0.040 | 0.021 | 0.035 | 1.000 | 0.006 | 0.006 | -0.031 | 0.029 | -0.043 | 0.034 | 0.034 | -0.033 | -0.019 | 0.046 | 0.034 | 0.007 | -0.041 | 0.011 | -0.001 | -0.039 | -0.028 | -0.038 |
| LAB | -0.009 | -0.030 | 0.053 | -0.052 | 0.016 | 0.006 | 1.000 | 0.072 | 0.075 | -0.065 | 0.039 | 0.007 | 0.055 | 0.016 | 0.009 | 0.017 | 0.003 | 0.048 | 0.003 | 0.006 | 0.022 | 0.016 | 0.000 | -0.056 |
| DGB | 0.015 | 0.026 | -0.005 | -0.006 | -0.018 | 0.006 | 0.072 | 1.000 | -0.021 | 0.014 | -0.027 | 0.055 | -0.049 | -0.019 | 0.043 | 0.026 | -0.034 | 0.064 | -0.005 | 0.005 | 0.055 | -0.109 | -0.065 | 0.003 |
| EPIC | 0.006 | 0.029 | 0.037 | -0.011 | -0.055 | -0.031 | 0.075 | -0.021 | 1.000 | -0.006 | 0.018 | 0.016 | -0.006 | 0.008 | 0.009 | 0.034 | -0.021 | -0.016 | -0.042 | -0.002 | -0.073 | -0.074 | 0.049 | -0.049 |
| BROCCOLIF3B | 0.051 | -0.046 | -0.031 | -0.015 | -0.023 | 0.029 | -0.065 | 0.014 | -0.006 | 1.000 | 0.003 | 0.029 | 0.026 | 0.006 | 0.018 | -0.006 | -0.021 | 0.022 | 0.006 | -0.028 | -0.029 | 0.053 | 0.019 | 0.013 |
| ALLO | 0.071 | 0.041 | 0.032 | -0.027 | -0.041 | -0.043 | 0.039 | -0.027 | 0.018 | 0.003 | 1.000 | 0.006 | 0.009 | -0.043 | -0.027 | -0.004 | -0.049 | -0.050 | -0.006 | -0.115 | -0.010 | -0.028 | -0.015 | -0.028 |
| BR | 0.039 | 0.005 | -0.006 | -0.065 | 0.045 | 0.034 | 0.007 | 0.055 | 0.016 | 0.029 | 0.006 | 1.000 | 0.016 | -0.011 | 0.000 | 0.018 | 0.026 | -0.022 | -0.017 | -0.005 | 0.047 | -0.007 | -0.014 | 0.041 |
| M | 0.028 | 0.004 | 0.021 | 0.001 | -0.012 | 0.034 | 0.055 | -0.049 | -0.006 | 0.026 | 0.009 | 0.016 | 1.000 | 0.005 | -0.027 | -0.019 | -0.027 | -0.001 | 0.020 | -0.012 | 0.053 | 0.042 | -0.000 | -0.006 |
| PIVX | -0.102 | -0.014 | -0.001 | 0.031 | -0.018 | -0.033 | 0.016 | -0.019 | 0.008 | 0.006 | -0.043 | -0.011 | 0.005 | 1.000 | -0.047 | -0.035 | -0.030 | -0.016 | -0.034 | 0.052 | -0.002 | 0.002 | 0.003 | -0.022 |
| BTW | 0.048 | -0.056 | 0.038 | 0.071 | -0.009 | -0.019 | 0.009 | 0.043 | 0.009 | 0.018 | -0.027 | 0.000 | -0.027 | -0.047 | 1.000 | 0.022 | -0.016 | 0.007 | -0.029 | 0.022 | -0.041 | -0.022 | -0.010 | 0.058 |
| NAORIS | 0.092 | 0.014 | -0.036 | 0.014 | 0.033 | 0.046 | 0.017 | 0.026 | 0.034 | -0.006 | -0.004 | 0.018 | -0.019 | -0.035 | 0.022 | 1.000 | 0.017 | 0.005 | -0.030 | -0.024 | 0.016 | -0.013 | 0.043 | 0.012 |
| NOW | -0.009 | 0.017 | 0.024 | -0.006 | 0.014 | 0.034 | 0.003 | -0.034 | -0.021 | -0.021 | -0.049 | 0.026 | -0.027 | -0.030 | -0.016 | 0.017 | 1.000 | -0.034 | -0.067 | -0.000 | -0.044 | 0.018 | 0.110 | -0.007 |
| PROM | -0.006 | 0.094 | 0.029 | 0.057 | -0.042 | 0.007 | 0.048 | 0.064 | -0.016 | 0.022 | -0.050 | -0.022 | -0.001 | -0.016 | 0.007 | 0.005 | -0.034 | 1.000 | -0.007 | -0.005 | 0.094 | -0.041 | -0.063 | 0.015 |
| GENIUS | 0.062 | 0.044 | 0.059 | -0.055 | 0.051 | -0.041 | 0.003 | -0.005 | -0.042 | 0.006 | -0.006 | -0.017 | 0.020 | -0.034 | -0.029 | -0.030 | -0.067 | -0.007 | 1.000 | 0.022 | 0.035 | 0.001 | -0.002 | 0.028 |
| WEN | 0.058 | -0.048 | 0.006 | 0.005 | 0.020 | 0.011 | 0.006 | 0.005 | -0.002 | -0.028 | -0.115 | -0.005 | -0.012 | 0.052 | 0.022 | -0.024 | -0.000 | -0.005 | 0.022 | 1.000 | -0.003 | 0.057 | 0.026 | 0.005 |
| QI | -0.013 | 0.026 | 0.003 | 0.046 | -0.030 | -0.001 | 0.022 | 0.055 | -0.073 | -0.029 | -0.010 | 0.047 | 0.053 | -0.002 | -0.041 | 0.016 | -0.044 | 0.094 | 0.035 | -0.003 | 1.000 | -0.044 | -0.022 | 0.005 |
| GUA | -0.034 | 0.032 | 0.032 | -0.011 | -0.002 | -0.039 | 0.016 | -0.109 | -0.074 | 0.053 | -0.028 | -0.007 | 0.042 | 0.002 | -0.022 | -0.013 | 0.018 | -0.041 | 0.001 | 0.057 | -0.044 | 1.000 | -0.035 | 0.001 |
| WET | 0.087 | -0.040 | -0.014 | -0.035 | -0.030 | -0.028 | 0.000 | -0.065 | 0.049 | 0.019 | -0.015 | -0.014 | -0.000 | 0.003 | -0.010 | 0.043 | 0.110 | -0.063 | -0.002 | 0.026 | -0.022 | -0.035 | 1.000 | -0.009 |
| STABLE | 0.078 | -0.104 | 0.039 | 0.019 | -0.004 | -0.038 | -0.056 | 0.003 | -0.049 | 0.013 | -0.028 | 0.041 | -0.006 | -0.022 | 0.058 | 0.012 | -0.007 | 0.015 | 0.028 | 0.005 | 0.005 | 0.001 | -0.009 | 1.000 |

The complete 232 × 232 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| HIMS | HIMSUSDT | 68.3% | 56.3% | WMTUSDT | -0.390 |
| APE | APEUSDT | 68.4% | 56.2% | BTCUSDT | 0.350 |
| BANANAS31 | BANANAS31USDT | 68.4% | 56.2% | BTCUSDT | 0.418 |
| HFT | HFTUSDT | 68.6% | 56.0% | ZORAUSDT | 0.238 |
| B | BUSDT | 68.7% | 56.0% | NFLXUSDT | 0.211 |
| TRX | TRXUSDT | 68.7% | 55.9% | BTCUSDT | 0.270 |
| DCR | DCRUSDT | 68.8% | 55.9% | BTCUSDT | 0.354 |
| GLM | GLMUSDT | 68.9% | 55.8% | BTCUSDT | 0.308 |
| AT | ATUSDT | 68.9% | 55.8% | BTCUSDT | 0.203 |
| MUBARAK | MUBARAKUSDT | 69.1% | 55.6% | BTCUSDT | 0.327 |
| SPY | SPYBUSDT | 69.1% | 55.6% | BTCUSDT | 0.265 |
| SQD | SQDUSDT | 69.1% | 55.6% | BTCUSDT | 0.235 |
| LLY | LLYUSDT | 69.2% | 55.5% | SPCXBUSDT | 0.204 |
| MOCA | MOCAUSDT | 69.3% | 55.4% | BTCUSDT | 0.333 |
| FTT | FTTUSDT | 69.3% | 55.4% | XPLUSDT | 0.180 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

