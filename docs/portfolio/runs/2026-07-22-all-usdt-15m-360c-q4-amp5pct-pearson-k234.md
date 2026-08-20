# Binance portfolio basis

Generated 2026-07-23T19:17:20.572Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-19T06:00Z through 2026-07-22T23:45Z
- Sampling: exactly 360 15m log returns (3.8 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 5.0% of the maximum unexplained variance, select the largest mean absolute 15m return
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
| 5 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.2% | 0.085 | BANKUSDT | 161.80 bp | 100.0% | 0 | 215.7M | 431.8% |
| 6 | ON | ONUSDT (usdm-futures) | usdm-futures | 99.2% | 0.098 | BANKUSDT | 158.92 bp | 100.0% | 0 | 25.3M | 436.8% |
| 7 | ACE | ACEUSDT (spot) | spot, usdm-futures | 99.5% | 0.064 | ONUSDT | 155.73 bp | 100.0% | 0.5 | 6.8M | 449.4% |
| 8 | PROM | PROMUSDT (spot) | spot, usdm-futures | 98.7% | 0.101 | ONUSDT | 138.72 bp | 100.0% | 0.5 | 3.8M | 395.7% |
| 9 | TLM | TLMUSDT (spot) | spot, usdm-futures | 98.7% | 0.128 | ACEUSDT | 135.17 bp | 100.0% | 0.3 | 7.7M | 395.4% |
| 10 | XNO | XNOUSDT (spot) | spot | 99.2% | 0.068 | BTCUSDT | 134.86 bp | 100.0% | 0.8 | 1.2M | 524.6% |
| 11 | ERA | ERAUSDT (spot) | spot, usdm-futures | 97.4% | 0.177 | BTCUSDT | 133.58 bp | 100.0% | 0.8 | 10.4M | 514.5% |
| 12 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 98.4% | 0.116 | ACEUSDT | 133.47 bp | 100.0% | 0.3 | 27.3M | 420.0% |
| 13 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 97.3% | 0.128 | AKEUSDT | 118.70 bp | 100.0% | 0.3 | 14.3M | 309.4% |
| 14 | LAB | LABUSDT (usdm-futures) | usdm-futures | 98.7% | 0.113 | AVAAIUSDT | 116.71 bp | 100.0% | 0.5 | 79M | 357.6% |
| 15 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 98.4% | 0.103 | BANKUSDT | 98.05 bp | 100.0% | 0.3 | 8.3M | 355.2% |
| 16 | DODO | DODOUSDT (spot) | spot | 97.0% | 0.177 | LABUSDT | 97.53 bp | 100.0% | 0.3 | 1.7M | 285.6% |
| 17 | BTTC | BTTCUSDT (spot) | spot | 97.1% | 0.128 | LABUSDT | 93.95 bp | 100.0% | 8.5 | 124.6K | 346.5% |
| 18 | ONE | ONEUSDT (spot) | spot, usdm-futures | 98.9% | 0.099 | LABUSDT | 93.36 bp | 100.0% | 2.5 | 1.9M | 444.9% |
| 19 | B | BUSDT (usdm-futures) | usdm-futures | 96.3% | 0.132 | ACEUSDT | 116.98 bp | 100.0% | 0.3 | 51.6M | 355.4% |
| 20 | DGB | DGBUSDT (spot) | spot | 98.1% | 0.074 | ONUSDT | 92.45 bp | 100.0% | 1 | 503.6K | 280.2% |
| 21 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 97.9% | 0.098 | DGBUSDT | 87.37 bp | 100.0% | 0.5 | 10.2M | 269.6% |
| 22 | PYR | PYRUSDT (spot) | spot | 96.9% | 0.138 | DODOUSDT | 83.87 bp | 100.0% | 1 | 1M | 216.0% |
| 23 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 96.2% | 0.140 | BANKUSDT | 75.39 bp | 100.0% | 1.3 | 6.6M | 212.1% |
| 24 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.4% | 0.130 | PYRUSDT | 73.08 bp | 100.0% | 0.8 | 6M | 189.3% |
| 25 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 97.7% | 0.127 | TLMUSDT | 70.60 bp | 100.0% | 1 | 2.9M | 225.9% |
| 26 | ZHIPU | ZHIPUUSDT (usdm-futures) | usdm-futures | 96.1% | 0.123 | ACEUSDT | 69.71 bp | 100.0% | 0.3 | 62.7M | 246.9% |
| 27 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 96.8% | 0.094 | BLESSUSDT | 68.09 bp | 100.0% | 0.3 | 8.5M | 188.9% |
| 28 | YB | YBUSDT (spot) | spot, usdm-futures | 97.8% | 0.089 | XNOUSDT | 66.29 bp | 100.0% | 0.8 | 1.3M | 182.0% |
| 29 | RE | REUSDT (spot) | spot, usdm-futures | 96.0% | 0.122 | HOMEUSDT | 66.14 bp | 100.0% | 0.3 | 57.9M | 166.9% |
| 30 | B2 | B2USDT (usdm-futures) | usdm-futures | 97.1% | 0.113 | PYRUSDT | 62.67 bp | 100.0% | 0.3 | 629.5K | 562.8% |
| 31 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 96.4% | 0.108 | TLMUSDT | 60.15 bp | 100.0% | 0.3 | 10.1M | 174.5% |
| 32 | FWDI | FWDIUSDT (usdm-futures) | usdm-futures | 96.6% | 0.139 | ONEUSDT | 59.39 bp | 100.0% | 0.8 | 1.8M | 280.3% |
| 33 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 96.0% | 0.115 | ACEUSDT | 54.91 bp | 100.0% | 0.5 | 3.9M | 151.4% |
| 34 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 96.4% | 0.105 | BTTCUSDT | 53.73 bp | 100.0% | 0.3 | 970K | 205.7% |
| 35 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 95.8% | 0.097 | BULLAUSDT | 53.41 bp | 100.0% | 0.8 | 1.6M | 139.3% |
| 36 | MET | METUSDT (spot) | spot, usdm-futures | 96.5% | 0.129 | BTCUSDT | 53.09 bp | 100.0% | 0.3 | 1.5M | 137.3% |
| 37 | PIVX | PIVXUSDT (spot) | spot | 96.9% | 0.102 | BTCUSDT | 48.79 bp | 100.0% | 1.3 | 250.8K | 203.7% |
| 38 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 95.6% | 0.109 | DGBUSDT | 51.24 bp | 100.0% | 0.3 | 2M | 156.2% |
| 39 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 96.4% | 0.093 | DGBUSDT | 46.27 bp | 100.0% | 0.3 | 4.1M | 151.6% |
| 40 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 95.9% | 0.106 | BTCUSDT | 44.23 bp | 100.0% | 0.3 | 1.8M | 115.7% |
| 41 | SKL | SKLUSDT (spot) | spot, usdm-futures | 95.4% | 0.156 | BTCUSDT | 44.12 bp | 100.0% | 1.3 | 908.4K | 160.3% |
| 42 | WET | WETUSDT (usdm-futures) | usdm-futures | 95.6% | 0.136 | PLAYUSDT | 30.92 bp | 100.0% | 0.3 | 1.3M | 83.2% |
| 43 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 95.7% | 0.109 | LABUSDT | 26.95 bp | 100.0% | 0.3 | 9.2M | 118.3% |
| 44 | O | OUSDT (usdm-futures) | usdm-futures | 95.1% | 0.106 | XNOUSDT | 44.88 bp | 100.0% | 0.3 | 4.8M | 138.8% |
| 45 | OPN | OPNUSDT (spot) | spot, usdm-futures | 95.1% | 0.115 | BTCUSDT | 43.52 bp | 100.0% | 1.3 | 32.3M | 122.3% |
| 46 | MMT | MMTUSDT (spot) | spot, usdm-futures | 94.9% | 0.119 | SKLUSDT | 35.47 bp | 100.0% | 0.5 | 667.7K | 90.2% |
| 47 | RARE | RAREUSDT (spot) | spot, usdm-futures | 94.7% | 0.157 | BTCUSDT | 25.37 bp | 100.0% | 2.8 | 176K | 82.7% |
| 48 | BABA | BABABUSDT (spot) | spot, usdm-futures | 95.3% | 0.087 | OUSDT | 22.73 bp | 100.0% | 0.8 | 553.4K | 67.4% |
| 49 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 94.7% | 0.126 | DODOUSDT | 51.53 bp | 100.0% | 0.3 | 2.9M | 140.9% |
| 50 | DKNG | DKNGUSDT (usdm-futures) | usdm-futures | 94.7% | 0.124 | BLESSUSDT | 16.59 bp | 100.0% | 1 | 457.5K | 71.2% |
| 51 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 96.9% | 0.077 | BTCUSDT | 14.51 bp | 100.0% | 0.8 | 1.1M | 61.2% |
| 52 | RIF | RIFUSDT (spot) | spot, usdm-futures | 92.4% | 0.156 | DEXEUSDT | 134.04 bp | 100.0% | 0.5 | 3.7M | 418.0% |
| 53 | LA | LAUSDT (spot) | spot, usdm-futures | 91.9% | 0.269 | ERAUSDT | 69.86 bp | 100.0% | 0.8 | 1.8M | 244.1% |
| 54 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 92.1% | 0.170 | BTCUSDT | 61.55 bp | 100.0% | 0.5 | 4.3M | 161.7% |
| 55 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 92.0% | 0.188 | REUSDT | 59.28 bp | 100.0% | 0.3 | 5.5M | 148.6% |
| 56 | TREE | TREEUSDT (spot) | spot, usdm-futures | 92.5% | 0.137 | BTCUSDT | 58.47 bp | 100.0% | 1 | 2.2M | 238.4% |
| 57 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 91.9% | 0.173 | TOWNSUSDT | 54.85 bp | 100.0% | 0.3 | 3.6M | 188.5% |
| 58 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 91.7% | 0.110 | BTCUSDT | 53.45 bp | 100.0% | 2.3 | 88K | 160.2% |
| 59 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 91.3% | 0.116 | BUSDT | 56.99 bp | 100.0% | 0.3 | 5.1M | 173.0% |
| 60 | TUT | TUTUSDT (spot) | spot, usdm-futures | 92.4% | 0.114 | TLMUSDT | 51.41 bp | 100.0% | 0.3 | 887.2K | 131.4% |
| 61 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 92.2% | 0.135 | GUAUSDT | 51.03 bp | 100.0% | 0.3 | 2.6M | 150.4% |
| 62 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 90.9% | 0.155 | TREEUSDT | 55.36 bp | 100.0% | 0.3 | 2.1M | 147.8% |
| 63 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 90.8% | 0.134 | TLMUSDT | 65.45 bp | 100.0% | 0.5 | 5.8M | 187.3% |
| 64 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 91.7% | 0.161 | BTCUSDT | 50.58 bp | 100.0% | 0.5 | 6.8M | 139.8% |
| 65 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 91.7% | 0.107 | ZHIPUUSDT | 47.43 bp | 100.0% | 0.5 | 2.2M | 128.1% |
| 66 | BR | BRUSDT (usdm-futures) | usdm-futures | 91.0% | 0.119 | OUSDT | 44.94 bp | 100.0% | 0 | 1.4M | 134.9% |
| 67 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 91.7% | 0.113 | AVAAIUSDT | 42.47 bp | 100.0% | 0.5 | 566.2K | 113.9% |
| 68 | ZIL | ZILUSDT (spot) | spot, usdm-futures | 90.4% | 0.178 | BTCUSDT | 38.85 bp | 100.0% | 1.3 | 1.2M | 120.8% |
| 69 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 90.4% | 0.158 | BTCUSDT | 38.42 bp | 100.0% | 0.5 | 897.8K | 105.5% |
| 70 | LISTA | LISTAUSDT (spot) | spot, usdm-futures | 90.7% | 0.203 | PIVXUSDT | 38.15 bp | 100.0% | 2.3 | 212.8K | 199.2% |
| 71 | QI | QIUSDT (spot) | spot | 90.9% | 0.112 | BTTCUSDT | 36.52 bp | 100.0% | 3.5 | 124.9K | 103.4% |
| 72 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 91.0% | 0.151 | BTCUSDT | 34.78 bp | 100.0% | 0.5 | 1.2M | 100.3% |
| 73 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 90.5% | 0.153 | BTCUSDT | 32.55 bp | 100.0% | 0.3 | 4.3M | 110.6% |
| 74 | M | MUSDT (usdm-futures) | usdm-futures | 90.5% | 0.132 | PYRUSDT | 28.94 bp | 100.0% | 0.3 | 1.9M | 74.6% |
| 75 | REQ | REQUSDT (spot) | spot | 92.4% | 0.232 | BTCUSDT | 26.91 bp | 100.0% | 1.5 | 220.3K | 127.6% |
| 76 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 89.5% | 0.132 | TRUTHUSDT | 37.65 bp | 100.0% | 0.3 | 962.6K | 93.7% |
| 77 | QUICK | QUICKUSDT (spot) | spot | 89.5% | 0.164 | ESPORTSUSDT | 33.27 bp | 100.0% | 0.8 | 41.2K | 83.0% |
| 78 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 88.9% | 0.140 | LUMIAUSDT | 33.89 bp | 100.0% | 1.3 | 2M | 98.8% |
| 79 | BEL | BELUSDT (spot) | spot, usdm-futures | 89.2% | 0.170 | BTCUSDT | 32.13 bp | 100.0% | 0.8 | 427K | 81.7% |
| 80 | TA | TAUSDT (usdm-futures) | usdm-futures | 89.0% | 0.130 | PYRUSDT | 29.86 bp | 100.0% | 0.5 | 1.7M | 84.1% |
| 81 | Q | QUSDT (usdm-futures) | usdm-futures | 90.4% | 0.125 | BLESSUSDT | 25.93 bp | 100.0% | 0.3 | 848K | 82.4% |
| 82 | AMP | AMPUSDT (spot) | spot | 89.9% | 0.144 | B2USDT | 23.47 bp | 100.0% | 1.5 | 383.3K | 70.6% |
| 83 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 88.4% | 0.145 | BRUSDT | 42.56 bp | 100.0% | 0.3 | 4.3M | 127.6% |
| 84 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 88.3% | 0.133 | BTCUSDT | 46.83 bp | 100.0% | 1.5 | 1.6M | 133.9% |
| 85 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 89.2% | 0.159 | TAGUSDT | 23.39 bp | 100.0% | 1 | 255.3K | 77.5% |
| 86 | SAGA | SAGAUSDT (spot) | spot, usdm-futures | 87.9% | 0.171 | BTCUSDT | 41.29 bp | 100.0% | 0.5 | 1.2M | 121.7% |
| 87 | ATM | ATMUSDT (spot) | spot | 87.2% | 0.180 | DEXEUSDT | 63.13 bp | 100.0% | 0.5 | 1.1M | 332.4% |
| 88 | RPL | RPLUSDT (spot) | spot, usdm-futures | 87.4% | 0.183 | BTCUSDT | 40.71 bp | 100.0% | 1.3 | 226.1K | 107.1% |
| 89 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 86.7% | 0.185 | WETUSDT | 35.08 bp | 100.0% | 0.8 | 2.4M | 110.7% |
| 90 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 86.6% | 0.157 | AIOTUSDT | 77.06 bp | 100.0% | 0.3 | 19M | 224.6% |
| 91 | TST | TSTUSDT (spot) | spot, usdm-futures | 86.6% | 0.141 | BTCUSDT | 38.12 bp | 100.0% | 0.5 | 320.3K | 103.4% |
| 92 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 86.8% | 0.154 | STBLUSDT | 30.62 bp | 100.0% | 0.8 | 1.5M | 80.5% |
| 93 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 86.6% | 0.161 | FIGHTUSDT | 28.99 bp | 100.0% | 0.5 | 3M | 78.6% |
| 94 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 88.2% | 0.163 | FWDIUSDT | 20.68 bp | 100.0% | 0.5 | 1.5M | 60.6% |
| 95 | CC | CCUSDT (usdm-futures) | usdm-futures | 86.0% | 0.170 | BTCUSDT | 17.00 bp | 100.0% | 0.3 | 2.6M | 44.7% |
| 96 | BAN | BANUSDT (usdm-futures) | usdm-futures | 86.0% | 0.176 | AMPUSDT | 22.81 bp | 100.0% | 0.5 | 1.3M | 60.8% |
| 97 | AIN | AINUSDT (usdm-futures) | usdm-futures | 85.6% | 0.156 | LIGHTUSDT | 52.05 bp | 100.0% | 0.3 | 1.2M | 151.4% |
| 98 | GTC | GTCUSDT (spot) | spot, usdm-futures | 85.6% | 0.146 | REUSDT | 51.17 bp | 100.0% | 2.8 | 151.5K | 158.8% |
| 99 | WEN | WENUSDT (usdm-futures) | usdm-futures | 87.2% | 0.148 | BLESSUSDT | 16.77 bp | 100.0% | 1.5 | 474.4K | 45.8% |
| 100 | KAVA | KAVAUSDT (spot) | spot, usdm-futures | 87.4% | 0.168 | BTCUSDT | 15.32 bp | 100.0% | 0.8 | 322.4K | 62.0% |
| 101 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 86.0% | 0.146 | DKNGUSDT | 11.84 bp | 100.0% | 1 | 18.6M | 31.9% |
| 102 | SUN | SUNUSDT (spot) | spot, usdm-futures | 85.8% | 0.236 | BTCUSDT | 8.30 bp | 100.0% | 1.5 | 338.7K | 23.7% |
| 103 | TAC | TACUSDT (usdm-futures) | usdm-futures | 84.6% | 0.152 | MUSDT | 73.51 bp | 100.0% | 0.3 | 4.6M | 184.9% |
| 104 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 84.5% | 0.153 | EVAAUSDT | 38.73 bp | 100.0% | 0.3 | 10.2M | 100.5% |
| 105 | POLYX | POLYXUSDT (spot) | spot, usdm-futures | 84.4% | 0.193 | BTCUSDT | 30.43 bp | 100.0% | 1.5 | 360.6K | 87.4% |
| 106 | AI | AIUSDT (spot) | spot | 84.1% | 0.151 | SAGAUSDT | 24.38 bp | 100.0% | 3.5 | 234.2K | 71.1% |
| 107 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 84.0% | 0.216 | VELVETUSDT | 74.96 bp | 100.0% | 0.3 | 2.6M | 303.1% |
| 108 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 83.8% | 0.130 | 币安人生USDT | 58.81 bp | 100.0% | 1 | 5.4M | 153.1% |
| 109 | U | UUSDT (spot) | spot | 85.8% | 0.138 | TLMUSDT | 0.47 bp | 100.0% | 3.8 | 15.3M | 1.3% |
| 110 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 82.8% | 0.147 | ESPORTSUSDT | 77.47 bp | 100.0% | 0.3 | 4.4M | 246.6% |
| 111 | PORTO | PORTOUSDT (spot) | spot | 81.9% | 0.186 | CAPUSDT | 63.49 bp | 100.0% | 0.8 | 544.6K | 258.7% |
| 112 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 82.3% | 0.152 | REUSDT | 57.90 bp | 100.0% | 0.3 | 4.1M | 150.6% |
| 113 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 82.1% | 0.292 | BLESSUSDT | 54.05 bp | 100.0% | 0.3 | 556.6K | 241.4% |
| 114 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 81.3% | 0.177 | TOWNSUSDT | 52.64 bp | 100.0% | 0.3 | 3.9M | 164.1% |
| 115 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 81.2% | 0.204 | BANKUSDT | 49.92 bp | 100.0% | 0.3 | 2.1M | 140.0% |
| 116 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 82.5% | 0.180 | AIOTUSDT | 46.46 bp | 100.0% | 0.3 | 1.7M | 124.4% |
| 117 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 81.1% | 0.193 | BTCUSDT | 43.75 bp | 100.0% | 0.3 | 7.6M | 108.8% |
| 118 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 81.1% | 0.178 | BTCUSDT | 42.63 bp | 100.0% | 0.3 | 1.2M | 111.7% |
| 119 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 81.5% | 0.135 | EPICUSDT | 41.51 bp | 100.0% | 0.3 | 1.7M | 106.1% |
| 120 | SNX | SNXUSDT (spot) | spot, usdm-futures | 81.5% | 0.226 | BTCUSDT | 40.34 bp | 100.0% | 2.3 | 1.3M | 113.6% |
| 121 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 80.7% | 0.231 | TREEUSDT | 38.97 bp | 100.0% | 0.8 | 791.1K | 98.8% |
| 122 | TURTLE | TURTLEUSDT (spot) | spot, usdm-futures | 80.9% | 0.221 | LABUSDT | 38.66 bp | 100.0% | 1.8 | 508K | 138.4% |
| 123 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 80.9% | 0.173 | LYNUSDT | 38.40 bp | 100.0% | 0.5 | 1.1M | 103.5% |
| 124 | BAS | BASUSDT (usdm-futures) | usdm-futures | 80.1% | 0.190 | REUSDT | 42.16 bp | 100.0% | 0.3 | 2.3M | 110.1% |
| 125 | ZORA | ZORAUSDT (usdm-futures) | usdm-futures | 80.2% | 0.234 | BTCUSDT | 31.95 bp | 100.0% | 0.3 | 2.2M | 89.1% |
| 126 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 80.7% | 0.301 | BTCUSDT | 31.58 bp | 100.0% | 0.5 | 325.4K | 86.2% |
| 127 | FLNC | FLNCUSDT (usdm-futures) | usdm-futures | 79.4% | 0.200 | ONEUSDT | 31.74 bp | 100.0% | 0.8 | 3.6M | 102.9% |
| 128 | THE | THEUSDT (spot) | spot, usdm-futures | 79.3% | 0.219 | BTCUSDT | 36.69 bp | 100.0% | 1 | 782.4K | 115.4% |
| 129 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 80.0% | 0.174 | FWDIUSDT | 29.71 bp | 100.0% | 0.8 | 1.9M | 82.0% |
| 130 | GEV | GEVUSDT (usdm-futures) | usdm-futures | 80.2% | 0.215 | RIFUSDT | 26.83 bp | 100.0% | 0.5 | 443.4K | 107.9% |
| 131 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 79.3% | 0.194 | BTCUSDT | 22.07 bp | 100.0% | 0.3 | 1.9M | 75.3% |
| 132 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 79.0% | 0.139 | OUSDT | 21.92 bp | 100.0% | 1.3 | 387.2K | 57.2% |
| 133 | IBM | IBMBUSDT (spot) | spot, usdm-futures | 79.3% | 0.207 | GOOGLBUSDT | 21.35 bp | 100.0% | 0.5 | 814.3K | 74.1% |
| 134 | ESP | ESPUSDT (spot) | spot, usdm-futures | 80.0% | 0.130 | RPLUSDT | 20.22 bp | 100.0% | 0.3 | 193.8K | 51.5% |
| 135 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 78.4% | 0.157 | BTCUSDT | 17.87 bp | 100.0% | 0.3 | 480.8K | 51.7% |
| 136 | IQ | IQUSDT (spot) | spot | 78.8% | 0.148 | BTCUSDT | 16.25 bp | 100.0% | 1 | 26.2K | 42.6% |
| 137 | BX | BXUSDT (usdm-futures) | usdm-futures | 79.7% | 0.176 | BTCUSDT | 15.84 bp | 100.0% | 0.3 | 869.3K | 43.8% |
| 138 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 80.2% | 0.143 | WETUSDT | 9.19 bp | 100.0% | 1.3 | 163K | 28.7% |
| 139 | SYN | SYNUSDT (spot) | spot, usdm-futures | 76.7% | 0.160 | VELVETUSDT | 75.92 bp | 100.0% | 0.3 | 4.5M | 215.4% |
| 140 | XEC | XECUSDT (spot) | spot, usdm-futures | 76.0% | 0.181 | LABUSDT | 68.77 bp | 100.0% | 0.5 | 2.7M | 185.5% |
| 141 | XAN | XANUSDT (usdm-futures) | usdm-futures | 75.8% | 0.162 | REQUSDT | 67.65 bp | 100.0% | 0.3 | 5.3M | 195.1% |
| 142 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 76.4% | 0.176 | PORTOUSDT | 59.55 bp | 100.0% | 0.3 | 2.8M | 167.7% |
| 143 | HEI | HEIUSDT (spot) | spot, usdm-futures | 75.7% | 0.186 | REUSDT | 54.42 bp | 100.0% | 0.5 | 1.6M | 142.4% |
| 144 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 75.5% | 0.235 | PLAYUSDT | 52.53 bp | 100.0% | 0.3 | 4.9M | 137.4% |
| 145 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 75.2% | 0.302 | B2USDT | 42.48 bp | 100.0% | 0.5 | 4.4M | 111.1% |
| 146 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 75.1% | 0.155 | ACUUSDT | 36.66 bp | 100.0% | 0.5 | 648.1K | 127.0% |
| 147 | VANA | VANAUSDT (spot) | spot, usdm-futures | 76.7% | 0.204 | REUSDT | 30.42 bp | 100.0% | 0.8 | 982.3K | 82.1% |
| 148 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 75.5% | 0.251 | BTCUSDT | 26.38 bp | 100.0% | 0.3 | 284K | 65.9% |
| 149 | WIN | WINUSDT (spot) | spot | 76.5% | 0.261 | BTCUSDT | 25.20 bp | 100.0% | 0.5 | 119K | 66.2% |
| 150 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 74.1% | 0.229 | VELVETUSDT | 59.00 bp | 100.0% | 0.3 | 17.4M | 156.7% |
| 151 | SOLV | SOLVUSDT (spot) | spot, usdm-futures | 73.6% | 0.245 | ESPORTSUSDT | 55.35 bp | 100.0% | 3.5 | 1M | 192.4% |
| 152 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 73.4% | 0.177 | REUSDT | 70.75 bp | 100.0% | 0.3 | 10.8M | 171.3% |
| 153 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 73.5% | 0.264 | LABUSDT | 46.52 bp | 100.0% | 0.3 | 14.5M | 121.8% |
| 154 | SMCI | SMCIUSDT (usdm-futures) | usdm-futures | 73.7% | 0.368 | FWDIUSDT | 38.71 bp | 100.0% | 0.5 | 6.9M | 218.8% |
| 155 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 73.3% | 0.180 | BTCUSDT | 37.83 bp | 100.0% | 0.5 | 2.5M | 99.2% |
| 156 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 73.1% | 0.170 | TAKEUSDT | 35.01 bp | 100.0% | 0.5 | 2.9M | 93.1% |
| 157 | CGPT | CGPTUSDT (spot) | spot, usdm-futures | 73.5% | 0.316 | BTCUSDT | 31.06 bp | 100.0% | 0.3 | 309.8K | 80.1% |
| 158 | TKO | TKOUSDT (spot) | spot | 73.4% | 0.223 | OUSDT | 30.88 bp | 100.0% | 1.3 | 150.6K | 97.3% |
| 159 | BAR | BARUSDT (spot) | spot | 73.0% | 0.190 | BABABUSDT | 30.24 bp | 100.0% | 2 | 564.3K | 85.7% |
| 160 | RAD | RADUSDT (spot) | spot | 73.7% | 0.226 | AIUSDT | 24.87 bp | 100.0% | 5 | 131.5K | 66.1% |
| 161 | APR | APRUSDT (usdm-futures) | usdm-futures | 73.1% | 0.170 | LISTAUSDT | 24.80 bp | 100.0% | 0.5 | 1.7M | 64.5% |
| 162 | CTSI | CTSIUSDT (spot) | spot, usdm-futures | 73.3% | 0.285 | BELUSDT | 22.80 bp | 100.0% | 0.8 | 251.2K | 82.2% |
| 163 | XPT | XPTUSDT (usdm-futures) | usdm-futures | 73.3% | 0.201 | BABABUSDT | 14.14 bp | 100.0% | 0.3 | 5.9M | 38.1% |
| 164 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 71.7% | 0.162 | BTCUSDT | 54.40 bp | 100.0% | 0.3 | 7M | 145.2% |
| 165 | SXT | SXTUSDT (spot) | spot, usdm-futures | 71.8% | 0.228 | BLESSUSDT | 36.50 bp | 100.0% | 0.5 | 790.8K | 97.0% |
| 166 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 71.6% | 0.246 | BTCUSDT | 36.14 bp | 100.0% | 0.3 | 2.6M | 101.9% |
| 167 | CAT | 1000CATUSDT (spot) | spot, usdm-futures | 70.8% | 0.268 | BTCUSDT | 34.35 bp | 100.0% | 2.3 | 79.2K | 102.9% |
| 168 | SCRT | SCRTUSDT (spot) | spot, usdm-futures | 71.0% | 0.228 | BTCUSDT | 32.13 bp | 100.0% | 1 | 217.7K | 88.5% |
| 169 | BIO | BIOUSDT (spot) | spot, usdm-futures | 71.6% | 0.326 | BTCUSDT | 31.66 bp | 100.0% | 0.5 | 1.4M | 89.1% |
| 170 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 70.6% | 0.350 | AMPUSDT | 26.27 bp | 100.0% | 0.8 | 642.4K | 78.9% |
| 171 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 71.0% | 0.265 | BTCUSDT | 23.52 bp | 100.0% | 0.3 | 707.1K | 62.9% |
| 172 | STRAX | STRAXUSDT (spot) | spot | 70.9% | 0.294 | BTCUSDT | 23.31 bp | 100.0% | 1 | 242.2K | 65.0% |
| 173 | NOW | NOWUSDT (usdm-futures) | usdm-futures | 70.4% | 0.292 | ONEUSDT | 21.51 bp | 100.0% | 0.5 | 934.8K | 75.1% |
| 174 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 69.9% | 0.251 | TACUSDT | 54.50 bp | 100.0% | 0.3 | 5.1M | 138.7% |
| 175 | NOK | NOKBUSDT (spot) | spot, usdm-futures | 69.9% | 0.229 | FLNCUSDT | 31.06 bp | 100.0% | 0.5 | 163.8K | 81.6% |
| 176 | JUV | JUVUSDT (spot) | spot | 70.1% | 0.153 | AIUSDT | 22.07 bp | 100.0% | 2 | 252.1K | 62.1% |
| 177 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 69.7% | 0.208 | BIRBUSDT | 30.44 bp | 100.0% | 0.3 | 798.5K | 83.1% |
| 178 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 69.6% | 0.230 | BTCUSDT | 28.72 bp | 100.0% | 0.5 | 1.1M | 72.2% |
| 179 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 69.8% | 0.368 | BTCUSDT | 19.81 bp | 100.0% | 1 | 116.6K | 55.4% |
| 180 | GNS | GNSUSDT (spot) | spot | 69.9% | 0.183 | ALLOUSDT | 16.98 bp | 100.0% | 2.5 | 73.7K | 57.5% |
| 181 | JST | JSTUSDT (spot) | spot, usdm-futures | 69.4% | 0.212 | GEVUSDT | 13.41 bp | 100.0% | 0.5 | 2.3M | 35.6% |
| 182 | IWM | IWMUSDT (usdm-futures) | usdm-futures | 70.9% | 0.289 | XPTUSDT | 8.74 bp | 100.0% | 1.3 | 378.4K | 36.1% |
| 183 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 67.6% | 0.203 | STRAXUSDT | 73.02 bp | 100.0% | 0.3 | 10.3M | 216.8% |
| 184 | NOM | NOMUSDT (spot) | spot, usdm-futures | 67.6% | 0.183 | BTCUSDT | 37.43 bp | 100.0% | 2.8 | 528.1K | 110.5% |
| 185 | ACT | ACTUSDT (spot) | spot, usdm-futures | 67.0% | 0.263 | KAVAUSDT | 40.81 bp | 100.0% | 0.5 | 669.2K | 116.9% |
| 186 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 67.2% | 0.277 | LIGHTUSDT | 34.33 bp | 100.0% | 0.3 | 649.6K | 91.4% |
| 187 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 67.6% | 0.223 | ZHIPUUSDT | 33.18 bp | 100.0% | 0.5 | 3.8M | 96.7% |
| 188 | WAL | WALUSDT (spot) | spot, usdm-futures | 67.6% | 0.365 | BTCUSDT | 28.98 bp | 100.0% | 1.3 | 298K | 75.6% |
| 189 | MINA | MINAUSDT (spot) | spot, usdm-futures | 68.0% | 0.294 | BTCUSDT | 24.44 bp | 100.0% | 1.8 | 249.7K | 65.4% |
| 190 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 67.1% | 0.309 | BTCUSDT | 17.62 bp | 100.0% | 0.8 | 118K | 46.8% |
| 191 | ACX | ACXUSDT (spot) | spot, usdm-futures | 68.4% | 0.228 | BTCUSDT | 11.05 bp | 100.0% | 1 | 55.1K | 32.4% |
| 192 | BONK | BONKUSDT (spot) | spot, usdm-futures | 64.8% | 0.310 | BTCUSDT | 53.34 bp | 100.0% | 1 | 3.7M | 141.4% |
| 193 | YGG | YGGUSDT (spot) | spot, usdm-futures | 65.0% | 0.343 | TREEUSDT | 47.53 bp | 100.0% | 0.5 | 1.4M | 140.9% |
| 194 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 65.3% | 0.239 | BTCUSDT | 46.98 bp | 100.0% | 0.3 | 9.7M | 119.4% |
| 195 | VVV | VVVUSDT (usdm-futures) | usdm-futures | 64.7% | 0.409 | BTCUSDT | 42.76 bp | 100.0% | 0.3 | 14.8M | 115.8% |
| 196 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 65.0% | 0.201 | GUAUSDT | 42.21 bp | 100.0% | 0.3 | 3M | 111.0% |
| 197 | PHA | PHAUSDT (spot) | spot, usdm-futures | 64.7% | 0.336 | BTCUSDT | 38.94 bp | 100.0% | 1 | 756.4K | 100.2% |
| 198 | RKLB | RKLBBUSDT (spot) | spot, usdm-futures | 64.4% | 0.231 | BTCUSDT | 37.83 bp | 100.0% | 0.3 | 148.9K | 106.4% |
| 199 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 64.4% | 0.246 | BTCUSDT | 34.06 bp | 100.0% | 0.5 | 689.9K | 89.1% |
| 200 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 65.2% | 0.249 | BTCUSDT | 32.71 bp | 100.0% | 0.3 | 1.2M | 91.2% |
| 201 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 65.1% | 0.247 | BTCUSDT | 32.56 bp | 100.0% | 0.5 | 547.2K | 83.5% |
| 202 | OPG | OPGUSDT (spot) | spot, usdm-futures | 64.6% | 0.252 | SCRTUSDT | 30.57 bp | 100.0% | 0.5 | 764.9K | 82.6% |
| 203 | BABY | BABYUSDT (spot) | spot, usdm-futures | 63.8% | 0.382 | BTCUSDT | 25.36 bp | 100.0% | 0.5 | 237K | 68.5% |
| 204 | HYPER | HYPERUSDT (spot) | spot, usdm-futures | 63.6% | 0.321 | BTCUSDT | 29.83 bp | 100.0% | 1.3 | 485.3K | 91.2% |
| 205 | ACM | ACMUSDT (spot) | spot | 65.0% | 0.187 | RPLUSDT | 24.35 bp | 100.0% | 1.8 | 210.3K | 62.9% |
| 206 | MBL | MBLUSDT (spot) | spot | 63.8% | 0.176 | BTCUSDT | 21.15 bp | 100.0% | 1 | 346.1K | 55.5% |
| 207 | MOVE | MOVEUSDT (spot) | spot, usdm-futures | 62.7% | 0.265 | BTCUSDT | 16.14 bp | 100.0% | 9.3 | 356.6K | 75.2% |
| 208 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 63.4% | 0.289 | BTCUSDT | 14.14 bp | 100.0% | 0.8 | 1.1M | 37.3% |
| 209 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 62.6% | 0.280 | BTCUSDT | 13.10 bp | 100.0% | 5 | 292.7K | 38.5% |
| 210 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 62.5% | 0.227 | DKNGUSDT | 11.15 bp | 100.0% | 0.8 | 320K | 33.6% |
| 211 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 63.8% | 0.175 | AINUSDT | 7.26 bp | 100.0% | 0.8 | 346.1K | 20.7% |
| 212 | C | CUSDT (spot) | spot, usdm-futures | 61.8% | 0.283 | BTCUSDT | 32.18 bp | 100.0% | 1 | 363.3K | 91.3% |
| 213 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 61.5% | 0.322 | BTCUSDT | 27.18 bp | 100.0% | 1 | 725.5K | 72.4% |
| 214 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 60.9% | 0.232 | MINAUSDT | 40.57 bp | 100.0% | 1 | 1.1M | 124.3% |
| 215 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 61.1% | 0.418 | BTCUSDT | 24.10 bp | 100.0% | 0.3 | 427.9K | 62.5% |
| 216 | TRX | TRXUSDT (spot) | spot, usdm-futures, coinm-futures | 62.1% | 0.270 | BTCUSDT | 4.32 bp | 100.0% | 1 | 21.2M | 12.0% |
| 217 | UB | UBUSDT (usdm-futures) | usdm-futures | 59.2% | 0.174 | VANAUSDT | 74.61 bp | 100.0% | 0.3 | 21.2M | 191.3% |
| 218 | STAR | STARUSDT (usdm-futures) | usdm-futures | 59.4% | 0.178 | ONEUSDT | 62.70 bp | 100.0% | 0.3 | 2.3M | 163.3% |
| 219 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 60.0% | 0.226 | EDGEUSDT | 58.32 bp | 100.0% | 0.3 | 4.7M | 149.1% |
| 220 | GLMR | GLMRUSDT (spot) | spot | 58.8% | 0.196 | IWMUSDT | 51.25 bp | 100.0% | 2 | 336.9K | 141.8% |
| 221 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 59.1% | 0.240 | BIOUSDT | 45.74 bp | 100.0% | 0.3 | 5.8M | 127.3% |
| 222 | PLUME | PLUMEUSDT (spot) | spot, usdm-futures | 59.1% | 0.271 | BTCUSDT | 44.05 bp | 100.0% | 0.5 | 1.1M | 109.1% |
| 223 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 58.9% | 0.233 | ERAUSDT | 43.14 bp | 100.0% | 0.3 | 1.5M | 137.5% |
| 224 | ORDER | ORDERUSDT (usdm-futures) | usdm-futures | 58.2% | 0.381 | TREEUSDT | 33.31 bp | 100.0% | 0.3 | 1.7M | 111.4% |
| 225 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 59.5% | 0.292 | LISTAUSDT | 32.82 bp | 100.0% | 0.5 | 534.3K | 85.9% |
| 226 | PUMP | PUMPUSDT (spot) | spot, usdm-futures | 57.5% | 0.251 | BTCUSDT | 48.30 bp | 100.0% | 0.5 | 10.4M | 128.8% |
| 227 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 57.1% | 0.220 | ZORAUSDT | 34.06 bp | 100.0% | 0.3 | 1.4M | 103.1% |
| 228 | DOOD | DOODUSDT (usdm-futures) | usdm-futures | 57.4% | 0.220 | BONKUSDT | 33.80 bp | 100.0% | 0.5 | 2.1M | 92.4% |
| 229 | FLOCK | FLOCKUSDT (usdm-futures) | usdm-futures | 57.1% | 0.323 | BTCUSDT | 31.40 bp | 100.0% | 0.3 | 1.9M | 85.0% |
| 230 | ENS | ENSUSDT (spot) | spot, usdm-futures | 57.5% | 0.401 | BTCUSDT | 28.69 bp | 100.0% | 0.8 | 1.2M | 72.8% |
| 231 | GPS | GPSUSDT (spot) | spot, usdm-futures | 56.9% | 0.160 | PRLUSDT | 26.49 bp | 100.0% | 0.8 | 261.9K | 71.0% |
| 232 | INIT | INITUSDT (spot) | spot, usdm-futures | 56.2% | 0.248 | BTCUSDT | 28.16 bp | 100.0% | 1.5 | 174.8K | 83.8% |
| 233 | XVS | XVSUSDT (spot) | spot, usdm-futures | 56.8% | 0.380 | BTCUSDT | 18.54 bp | 100.0% | 2.5 | 78.1K | 56.7% |
| 234 | SC | SCUSDT (spot) | spot | 55.8% | 0.348 | BTCUSDT | 16.42 bp | 100.0% | 1 | 63.1K | 41.3% |

## Diagnostics

- Basis size selected: 234
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.054
- Maximum pairwise absolute correlation: 0.418
- Mean whole-market projection R²: 84.9%
- Median whole-market projection R²: 80.2%
- 10th-percentile whole-market projection R²: 71.9%
- Minimum whole-market projection R²: 68.1%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.7% | 7.8% | 0.4% | 0.0% |
| 5 | 14.4% | 9.6% | 1.4% | 0.3% |
| 10 | 16.5% | 11.7% | 2.7% | 0.7% |
| 15 | 18.5% | 13.4% | 3.9% | 1.2% |
| 20 | 20.9% | 15.4% | 5.6% | 2.7% |
| 25 | 22.8% | 16.9% | 7.3% | 3.2% |
| 30 | 25.0% | 19.0% | 8.8% | 3.4% |
| 35 | 27.1% | 20.5% | 10.4% | 3.7% |
| 40 | 29.2% | 22.3% | 12.0% | 4.1% |
| 45 | 31.2% | 24.2% | 13.6% | 5.4% |
| 50 | 33.6% | 27.0% | 15.7% | 6.1% |
| 55 | 35.6% | 28.8% | 17.3% | 11.3% |
| 60 | 37.4% | 30.4% | 18.8% | 12.3% |
| 65 | 39.1% | 31.9% | 20.5% | 14.2% |
| 70 | 41.0% | 33.5% | 22.0% | 14.3% |
| 75 | 42.9% | 35.5% | 23.5% | 16.1% |
| 80 | 44.6% | 37.0% | 25.2% | 17.4% |
| 85 | 46.2% | 38.3% | 27.0% | 19.7% |
| 90 | 47.8% | 40.0% | 28.5% | 21.1% |
| 95 | 49.5% | 41.8% | 30.0% | 22.5% |
| 100 | 51.1% | 43.1% | 31.8% | 23.5% |
| 105 | 52.8% | 44.7% | 33.4% | 25.5% |
| 110 | 54.3% | 46.3% | 35.1% | 30.2% |
| 115 | 55.8% | 47.8% | 36.8% | 31.2% |
| 120 | 57.3% | 49.3% | 38.3% | 31.9% |
| 125 | 58.9% | 50.9% | 40.0% | 32.6% |
| 130 | 60.4% | 52.5% | 41.6% | 34.5% |
| 135 | 62.2% | 54.9% | 43.2% | 35.5% |
| 140 | 63.7% | 56.8% | 44.6% | 40.2% |
| 145 | 65.0% | 58.1% | 45.8% | 41.2% |
| 150 | 66.4% | 59.8% | 47.5% | 43.1% |
| 155 | 67.7% | 60.9% | 48.8% | 43.9% |
| 160 | 68.9% | 62.1% | 50.5% | 44.0% |
| 165 | 70.4% | 63.9% | 52.2% | 47.2% |
| 170 | 71.5% | 64.9% | 53.4% | 47.9% |
| 175 | 72.7% | 66.4% | 55.0% | 48.8% |
| 180 | 73.9% | 67.6% | 56.7% | 49.4% |
| 185 | 75.1% | 68.8% | 58.3% | 53.0% |
| 190 | 76.2% | 69.9% | 59.7% | 53.2% |
| 195 | 77.3% | 71.3% | 60.9% | 56.4% |
| 200 | 78.3% | 72.6% | 62.4% | 57.1% |
| 205 | 79.4% | 73.9% | 63.8% | 58.3% |
| 210 | 80.4% | 75.0% | 65.3% | 59.3% |
| 215 | 81.4% | 76.3% | 66.6% | 61.4% |
| 220 | 82.3% | 77.3% | 68.1% | 64.4% |
| 225 | 83.3% | 78.5% | 69.3% | 65.5% |
| 230 | 84.2% | 79.5% | 70.8% | 66.4% |
| 234 | 84.9% | 80.2% | 71.9% | 68.1% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | BANK | ESPORTS | DEXE | AKE | ON | ACE | PROM | TLM | XNO | ERA | AVAAI | BULLA | LAB | BLESS | DODO | BTTC | ONE | B | DGB | NIGHT | PYR | TOWNS | HOME |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.073 | 0.072 | -0.008 | 0.060 | -0.055 | 0.038 | -0.006 | -0.068 | -0.068 | 0.177 | -0.007 | -0.099 | -0.009 | -0.017 | 0.031 | 0.063 | -0.039 | 0.093 | 0.015 | 0.029 | -0.087 | 0.057 | -0.039 |
| BANK | -0.073 | 1.000 | -0.006 | 0.050 | 0.085 | 0.098 | -0.044 | 0.094 | -0.001 | -0.038 | -0.040 | 0.070 | 0.071 | -0.030 | 0.103 | 0.019 | -0.054 | 0.013 | 0.093 | 0.026 | 0.029 | -0.039 | 0.140 | 0.111 |
| ESPORTS | 0.072 | -0.006 | 1.000 | -0.061 | -0.052 | 0.045 | 0.030 | 0.029 | 0.002 | -0.040 | 0.015 | -0.077 | 0.052 | 0.053 | -0.015 | -0.029 | 0.049 | -0.067 | 0.026 | -0.005 | -0.008 | -0.027 | -0.057 | -0.020 |
| DEXE | -0.008 | 0.050 | -0.061 | 1.000 | -0.030 | -0.032 | -0.028 | 0.057 | 0.011 | 0.021 | 0.035 | 0.036 | -0.004 | -0.052 | -0.055 | 0.076 | -0.011 | 0.036 | -0.000 | -0.006 | 0.065 | 0.058 | 0.109 | 0.019 |
| AKE | 0.060 | 0.085 | -0.052 | -0.030 | 1.000 | 0.012 | 0.020 | -0.049 | 0.031 | -0.061 | 0.036 | 0.002 | 0.128 | 0.053 | 0.033 | -0.054 | 0.045 | 0.001 | -0.038 | -0.066 | 0.017 | 0.003 | 0.063 | 0.018 |
| ON | -0.055 | 0.098 | 0.045 | -0.032 | 0.012 | 1.000 | 0.064 | 0.101 | 0.039 | -0.048 | 0.015 | 0.026 | -0.074 | -0.007 | 0.039 | 0.014 | -0.095 | -0.046 | 0.066 | 0.074 | -0.062 | -0.023 | -0.000 | 0.033 |
| ACE | 0.038 | -0.044 | 0.030 | -0.028 | 0.020 | 0.064 | 1.000 | -0.042 | -0.128 | 0.035 | 0.110 | -0.116 | -0.065 | 0.016 | -0.028 | -0.096 | -0.009 | -0.019 | -0.132 | -0.018 | 0.007 | 0.054 | 0.072 | 0.071 |
| PROM | -0.006 | 0.094 | 0.029 | 0.057 | -0.049 | 0.101 | -0.042 | 1.000 | -0.035 | 0.007 | -0.013 | -0.060 | -0.054 | 0.048 | 0.060 | 0.051 | -0.001 | 0.017 | 0.095 | 0.064 | 0.080 | 0.038 | 0.071 | 0.011 |
| TLM | -0.068 | -0.001 | 0.002 | 0.011 | 0.031 | 0.039 | -0.128 | -0.035 | 1.000 | -0.008 | -0.023 | -0.018 | 0.046 | 0.018 | 0.041 | 0.025 | -0.048 | -0.026 | -0.012 | -0.068 | -0.014 | 0.065 | -0.005 | -0.023 |
| XNO | -0.068 | -0.038 | -0.040 | 0.021 | -0.061 | -0.048 | 0.035 | 0.007 | -0.008 | 1.000 | -0.081 | 0.006 | -0.010 | 0.006 | -0.031 | -0.048 | -0.012 | -0.021 | -0.039 | 0.006 | 0.033 | -0.044 | 0.011 | -0.013 |
| ERA | 0.177 | -0.040 | 0.015 | 0.035 | 0.036 | 0.015 | 0.110 | -0.013 | -0.023 | -0.081 | 1.000 | -0.008 | -0.063 | 0.038 | 0.081 | -0.055 | -0.013 | -0.004 | -0.073 | -0.053 | -0.006 | -0.015 | -0.027 | -0.005 |
| AVAAI | -0.007 | 0.070 | -0.077 | 0.036 | 0.002 | 0.026 | -0.116 | -0.060 | -0.018 | 0.006 | -0.008 | 1.000 | -0.027 | -0.113 | 0.001 | 0.087 | -0.124 | -0.007 | -0.054 | -0.044 | -0.016 | -0.040 | 0.020 | -0.031 |
| BULLA | -0.099 | 0.071 | 0.052 | -0.004 | 0.128 | -0.074 | -0.065 | -0.054 | 0.046 | -0.010 | -0.063 | -0.027 | 1.000 | -0.015 | -0.023 | -0.030 | 0.080 | -0.006 | -0.077 | 0.032 | -0.056 | 0.000 | -0.017 | -0.082 |
| LAB | -0.009 | -0.030 | 0.053 | -0.052 | 0.053 | -0.007 | 0.016 | 0.048 | 0.018 | 0.006 | 0.038 | -0.113 | -0.015 | 1.000 | 0.025 | -0.177 | 0.128 | -0.099 | 0.057 | 0.072 | 0.036 | -0.033 | -0.098 | 0.066 |
| BLESS | -0.017 | 0.103 | -0.015 | -0.055 | 0.033 | 0.039 | -0.028 | 0.060 | 0.041 | -0.031 | 0.081 | 0.001 | -0.023 | 0.025 | 1.000 | 0.009 | 0.045 | -0.005 | 0.022 | -0.014 | -0.036 | 0.092 | 0.065 | 0.097 |
| DODO | 0.031 | 0.019 | -0.029 | 0.076 | -0.054 | 0.014 | -0.096 | 0.051 | 0.025 | -0.048 | -0.055 | 0.087 | -0.030 | -0.177 | 0.009 | 1.000 | -0.008 | 0.037 | -0.015 | -0.031 | 0.048 | 0.138 | 0.019 | 0.018 |
| BTTC | 0.063 | -0.054 | 0.049 | -0.011 | 0.045 | -0.095 | -0.009 | -0.001 | -0.048 | -0.012 | -0.013 | -0.124 | 0.080 | 0.128 | 0.045 | -0.008 | 1.000 | -0.007 | -0.023 | -0.019 | 0.056 | -0.028 | -0.036 | 0.020 |
| ONE | -0.039 | 0.013 | -0.067 | 0.036 | 0.001 | -0.046 | -0.019 | 0.017 | -0.026 | -0.021 | -0.004 | -0.007 | -0.006 | -0.099 | -0.005 | 0.037 | -0.007 | 1.000 | -0.016 | -0.021 | 0.036 | 0.037 | 0.083 | 0.043 |
| B | 0.093 | 0.093 | 0.026 | -0.000 | -0.038 | 0.066 | -0.132 | 0.095 | -0.012 | -0.039 | -0.073 | -0.054 | -0.077 | 0.057 | 0.022 | -0.015 | -0.023 | -0.016 | 1.000 | -0.027 | -0.038 | -0.029 | 0.022 | 0.069 |
| DGB | 0.015 | 0.026 | -0.005 | -0.006 | -0.066 | 0.074 | -0.018 | 0.064 | -0.068 | 0.006 | -0.053 | -0.044 | 0.032 | 0.072 | -0.014 | -0.031 | -0.019 | -0.021 | -0.027 | 1.000 | -0.098 | 0.032 | -0.002 | 0.023 |
| NIGHT | 0.029 | 0.029 | -0.008 | 0.065 | 0.017 | -0.062 | 0.007 | 0.080 | -0.014 | 0.033 | -0.006 | -0.016 | -0.056 | 0.036 | -0.036 | 0.048 | 0.056 | 0.036 | -0.038 | -0.098 | 1.000 | 0.017 | -0.014 | 0.096 |
| PYR | -0.087 | -0.039 | -0.027 | 0.058 | 0.003 | -0.023 | 0.054 | 0.038 | 0.065 | -0.044 | -0.015 | -0.040 | 0.000 | -0.033 | 0.092 | 0.138 | -0.028 | 0.037 | -0.029 | 0.032 | 0.017 | 1.000 | -0.040 | 0.130 |
| TOWNS | 0.057 | 0.140 | -0.057 | 0.109 | 0.063 | -0.000 | 0.072 | 0.071 | -0.005 | 0.011 | -0.027 | 0.020 | -0.017 | -0.098 | 0.065 | 0.019 | -0.036 | 0.083 | 0.022 | -0.002 | -0.014 | -0.040 | 1.000 | 0.025 |
| HOME | -0.039 | 0.111 | -0.020 | 0.019 | 0.018 | 0.033 | 0.071 | 0.011 | -0.023 | -0.013 | -0.005 | -0.031 | -0.082 | 0.066 | 0.097 | 0.018 | 0.020 | 0.043 | 0.069 | 0.023 | 0.096 | 0.130 | 0.025 | 1.000 |

The complete 234 × 234 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| JPM | JPMUSDT | 68.1% | 56.5% | DKNGUSDT | -0.188 |
| MTL | MTLUSDT | 68.3% | 56.3% | BTCUSDT | 0.276 |
| WLFI | WLFIUSDT | 69.3% | 55.4% | B2USDT | 0.242 |
| ICX | ICXUSDT | 69.4% | 55.3% | BTCUSDT | 0.367 |
| OPENAI | OPENAIUSDT | 69.5% | 55.2% | ANTHROPICUSDT | 0.361 |
| AGLD | AGLDUSDT | 69.6% | 55.2% | BANANAS31USDT | 0.280 |
| XPL | XPLUSDT | 69.6% | 55.1% | BTCUSDT | 0.354 |
| XBI | XBIUSDT | 69.7% | 55.0% | BTCUSDT | 0.229 |
| CHILLGUY | CHILLGUYUSDT | 69.8% | 54.9% | BTCUSDT | 0.356 |
| ANKR | ANKRUSDT | 69.8% | 54.9% | ACEUSDT | 0.252 |
| XMR | XMRUSDT | 69.9% | 54.9% | BTCUSDT | 0.353 |
| KSM | KSMUSDT | 69.9% | 54.9% | BTCUSDT | 0.443 |
| V | VUSDT | 69.9% | 54.8% | ONEUSDT | -0.293 |
| SPY | SPYBUSDT | 70.0% | 54.8% | XPTUSDT | 0.272 |
| ZRO | ZROUSDT | 70.0% | 54.8% | BTCUSDT | 0.344 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

