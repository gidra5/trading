# Binance portfolio basis

Generated 2026-07-23T19:09:04.506Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-08T00:00Z through 2026-07-22T23:00Z
- Sampling: exactly 360 1h log returns (15 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 1.0% of the maximum unexplained variance, select the largest mean absolute 1h return
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 671 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 23.17 bp | 100.0% | 0 | 1.2B | 31.4% |
| 2 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 99.7% | 0.072 | BTCUSDT | 377.08 bp | 100.0% | 0 | 219.8M | 935.5% |
| 3 | BANK | BANKUSDT (spot) | spot, usdm-futures | 99.9% | 0.033 | BTCUSDT | 285.54 bp | 100.0% | 3 | 2.5M | 589.7% |
| 4 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.9% | 0.031 | EVAAUSDT | 282.37 bp | 100.0% | 1 | 185.8M | 479.2% |
| 5 | TLM | TLMUSDT (spot) | spot, usdm-futures | 99.6% | 0.080 | BANKUSDT | 202.82 bp | 100.0% | 1 | 4M | 315.8% |
| 6 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 99.6% | 0.083 | BTCUSDT | 178.55 bp | 100.0% | 1 | 13.6M | 243.7% |
| 7 | DODO | DODOUSDT (spot) | spot | 99.9% | 0.041 | TRIAUSDT | 172.04 bp | 100.0% | 1 | 3.3M | 319.7% |
| 8 | ON | ONUSDT (usdm-futures) | usdm-futures | 99.4% | 0.067 | TLMUSDT | 169.16 bp | 100.0% | 1 | 2.4M | 256.1% |
| 9 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 99.7% | 0.042 | EVAAUSDT | 161.97 bp | 100.0% | 1 | 21.1M | 282.4% |
| 10 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 99.2% | 0.081 | EDGEUSDT | 160.75 bp | 100.0% | 0 | 11.9M | 241.4% |
| 11 | PYR | PYRUSDT (spot) | spot | 99.3% | 0.076 | BTCUSDT | 151.47 bp | 100.0% | 3 | 1.1M | 267.1% |
| 12 | BTTC | BTTCUSDT (spot) | spot | 99.4% | 0.067 | BTCUSDT | 145.78 bp | 100.0% | 17 | 139.3K | 220.6% |
| 13 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 99.5% | 0.073 | BTCUSDT | 138.80 bp | 100.0% | 1 | 6.8M | 199.7% |
| 14 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 99.3% | 0.070 | AKEUSDT | 133.78 bp | 100.0% | 1 | 7.1M | 236.9% |
| 15 | XEC | XECUSDT (spot) | spot, usdm-futures | 98.9% | 0.066 | BLUAIUSDT | 132.06 bp | 100.0% | 2 | 2.4M | 218.4% |
| 16 | ATM | ATMUSDT (spot) | spot | 99.3% | 0.059 | BTCUSDT | 108.54 bp | 100.0% | 1 | 1.2M | 252.5% |
| 17 | XNO | XNOUSDT (spot) | spot | 98.5% | 0.103 | ONUSDT | 126.64 bp | 100.0% | 3 | 43.1K | 320.2% |
| 18 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 98.6% | 0.104 | BTTCUSDT | 104.10 bp | 100.0% | 1 | 1.4M | 230.9% |
| 19 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 98.3% | 0.088 | BANKUSDT | 86.69 bp | 100.0% | 1 | 3.5M | 142.0% |
| 20 | QUICK | QUICKUSDT (spot) | spot | 98.3% | 0.087 | BTCUSDT | 51.31 bp | 100.0% | 2 | 54.6K | 74.9% |
| 21 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 98.1% | 0.093 | TLMUSDT | 196.90 bp | 100.0% | 1 | 28.2M | 310.1% |
| 22 | AIN | AINUSDT (usdm-futures) | usdm-futures | 97.8% | 0.081 | BANKUSDT | 92.15 bp | 100.0% | 1 | 984.9K | 133.5% |
| 23 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 97.8% | 0.093 | BANKUSDT | 134.78 bp | 100.0% | 2 | 2.1M | 170.6% |
| 24 | AWE | AWEUSDT (spot) | spot, usdm-futures | 98.0% | 0.113 | BTCUSDT | 67.11 bp | 100.0% | 1 | 346.5K | 101.9% |
| 25 | AMP | AMPUSDT (spot) | spot | 98.0% | 0.110 | TRIAUSDT | 42.54 bp | 100.0% | 5 | 470.5K | 113.2% |
| 26 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 97.8% | 0.102 | BTCUSDT | 24.68 bp | 100.0% | 1 | 1.7M | 66.8% |
| 27 | MSFT | MSFTBUSDT (spot) | spot, usdm-futures | 97.7% | 0.138 | BTCUSDT | 22.08 bp | 100.0% | 1 | 147.6K | 33.7% |
| 28 | BAR | BARUSDT (spot) | spot | 97.5% | 0.117 | DODOUSDT | 63.05 bp | 100.0% | 3 | 473.1K | 115.6% |
| 29 | DIS | DISUSDT (usdm-futures) | usdm-futures | 97.4% | 0.091 | MAGMAUSDT | 17.83 bp | 100.0% | 2 | 218.7K | 26.0% |
| 30 | KGST | KGSTUSDT (spot) | spot | 97.8% | 0.106 | TRIAUSDT | 2.35 bp | 100.0% | 8 | 343.6K | 6.3% |
| 31 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 97.1% | 0.114 | BTCUSDT | 143.30 bp | 100.0% | 1 | 4.4M | 214.2% |
| 32 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 96.4% | 0.103 | BTCUSDT | 344.21 bp | 100.0% | 1 | 83.9M | 551.2% |
| 33 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 96.5% | 0.109 | TRIAUSDT | 142.66 bp | 100.0% | 1 | 20.8M | 191.7% |
| 34 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 96.6% | 0.101 | EVAAUSDT | 97.00 bp | 100.0% | 1 | 2.4M | 182.0% |
| 35 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 95.8% | 0.166 | PYRUSDT | 127.96 bp | 100.0% | 0 | 3.8M | 215.5% |
| 36 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 95.8% | 0.123 | EDGEUSDT | 88.69 bp | 100.0% | 1 | 2M | 129.7% |
| 37 | THE | THEUSDT (spot) | spot, usdm-futures | 95.7% | 0.153 | AINUSDT | 79.60 bp | 100.0% | 2 | 591.8K | 145.1% |
| 38 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 95.5% | 0.117 | BTCUSDT | 145.25 bp | 100.0% | 1 | 10.5M | 194.1% |
| 39 | B2 | B2USDT (usdm-futures) | usdm-futures | 95.6% | 0.113 | ANTHROPICUSDT | 70.67 bp | 100.0% | 1 | 916.5K | 123.3% |
| 40 | GLMR | GLMRUSDT (spot) | spot | 95.9% | 0.140 | ATMUSDT | 68.74 bp | 100.0% | 9 | 409.3K | 100.9% |
| 41 | QNTX | QNTXUSDT (usdm-futures) | usdm-futures | 95.2% | 0.110 | ALLOUSDT | 67.19 bp | 100.0% | 1 | 2.5M | 113.9% |
| 42 | BAN | BANUSDT (usdm-futures) | usdm-futures | 95.5% | 0.142 | BTCUSDT | 59.94 bp | 100.0% | 2 | 1.3M | 97.3% |
| 43 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 94.8% | 0.121 | TRADOORUSDT | 125.42 bp | 100.0% | 1 | 2.7M | 188.4% |
| 44 | MMT | MMTUSDT (spot) | spot, usdm-futures | 94.8% | 0.157 | EVAAUSDT | 100.40 bp | 100.0% | 1 | 1.2M | 134.6% |
| 45 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 94.6% | 0.147 | ANTHROPICUSDT | 86.81 bp | 100.0% | 1 | 1.9M | 135.9% |
| 46 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 94.8% | 0.153 | TRIAUSDT | 48.30 bp | 100.0% | 1 | 2.3M | 74.5% |
| 47 | QI | QIUSDT (spot) | spot | 94.1% | 0.105 | TRIAUSDT | 53.58 bp | 100.0% | 10 | 121.9K | 97.2% |
| 48 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 94.0% | 0.160 | EDGEUSDT | 168.34 bp | 100.0% | 1 | 7.4M | 298.4% |
| 49 | JST | JSTUSDT (spot) | spot, usdm-futures | 94.3% | 0.134 | ONUSDT | 35.11 bp | 100.0% | 0 | 3.2M | 45.2% |
| 50 | SYN | SYNUSDT (spot) | spot, usdm-futures | 93.3% | 0.155 | AERGOUSDT | 191.68 bp | 100.0% | 0 | 7.2M | 273.8% |
| 51 | B | BUSDT (usdm-futures) | usdm-futures | 92.9% | 0.193 | PYRUSDT | 234.77 bp | 100.0% | 1 | 24.3M | 400.1% |
| 52 | PROM | PROMUSDT (spot) | spot, usdm-futures | 92.9% | 0.209 | BANKUSDT | 116.68 bp | 100.0% | 2 | 300.4K | 211.7% |
| 53 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 92.8% | 0.129 | TRIAUSDT | 113.60 bp | 100.0% | 1 | 2.6M | 164.3% |
| 54 | ACE | ACEUSDT (spot) | spot, usdm-futures | 93.2% | 0.150 | DISUSDT | 110.95 bp | 100.0% | 2 | 170.2K | 251.9% |
| 55 | O | OUSDT (usdm-futures) | usdm-futures | 92.7% | 0.115 | BANUSDT | 95.88 bp | 100.0% | 2 | 9.1M | 145.8% |
| 56 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 92.9% | 0.184 | TRIAUSDT | 85.02 bp | 100.0% | 1 | 1.2M | 114.2% |
| 57 | TREE | TREEUSDT (spot) | spot, usdm-futures | 92.8% | 0.159 | ZESTUSDT | 83.47 bp | 100.0% | 3 | 6.7M | 160.3% |
| 58 | DGB | DGBUSDT (spot) | spot | 92.3% | 0.169 | BTTCUSDT | 131.76 bp | 100.0% | 3 | 114.2K | 229.6% |
| 59 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 92.2% | 0.121 | MMTUSDT | 56.42 bp | 100.0% | 2 | 612.4K | 72.8% |
| 60 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 92.5% | 0.132 | BTCUSDT | 53.40 bp | 100.0% | 3 | 801.5K | 125.1% |
| 61 | RIF | RIFUSDT (spot) | spot, usdm-futures | 91.7% | 0.208 | AERGOUSDT | 162.95 bp | 100.0% | 2 | 1.8M | 246.3% |
| 62 | SKL | SKLUSDT (spot) | spot, usdm-futures | 91.7% | 0.120 | ATMUSDT | 117.72 bp | 100.0% | 4 | 1.1M | 198.4% |
| 63 | OGN | OGNUSDT (spot) | spot, usdm-futures | 91.6% | 0.172 | GLMRUSDT | 75.37 bp | 100.0% | 1 | 1M | 163.3% |
| 64 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 91.1% | 0.122 | XECUSDT | 136.58 bp | 100.0% | 1 | 2.6M | 225.5% |
| 65 | AT | ATUSDT (spot) | spot, usdm-futures | 91.0% | 0.172 | BTCUSDT | 58.87 bp | 100.0% | 1 | 288.6K | 83.7% |
| 66 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 91.0% | 0.232 | BTCUSDT | 48.68 bp | 100.0% | 5 | 233.1K | 90.2% |
| 67 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 91.3% | 0.137 | MSFTBUSDT | 40.78 bp | 100.0% | 1 | 644K | 54.5% |
| 68 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 90.7% | 0.133 | BLESSUSDT | 154.47 bp | 100.0% | 2 | 12.7M | 195.4% |
| 69 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 90.1% | 0.157 | ESPORTSUSDT | 155.83 bp | 100.0% | 0 | 3.5M | 254.1% |
| 70 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 90.5% | 0.123 | EVAAUSDT | 97.91 bp | 100.0% | 1 | 2.2M | 137.2% |
| 71 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 90.2% | 0.203 | EVAAUSDT | 95.09 bp | 100.0% | 0 | 2.6M | 139.4% |
| 72 | CATI | CATIUSDT (spot) | spot, usdm-futures | 90.5% | 0.173 | BTCUSDT | 79.97 bp | 100.0% | 2 | 340.5K | 101.3% |
| 73 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 90.1% | 0.144 | BTCUSDT | 66.10 bp | 100.0% | 1 | 1.1M | 90.1% |
| 74 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 90.2% | 0.313 | BTCUSDT | 63.33 bp | 100.0% | 2 | 880.8K | 127.1% |
| 75 | M | MUSDT (usdm-futures) | usdm-futures | 89.5% | 0.159 | MSFTBUSDT | 100.16 bp | 100.0% | 1 | 3.2M | 165.0% |
| 76 | ONE | ONEUSDT (spot) | spot, usdm-futures | 89.1% | 0.215 | BTCUSDT | 87.85 bp | 100.0% | 5 | 171.3K | 191.9% |
| 77 | GNO | GNOUSDT (spot) | spot | 89.1% | 0.208 | AKEUSDT | 41.19 bp | 100.0% | 2 | 124.9K | 72.7% |
| 78 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 89.4% | 0.142 | DISUSDT | 26.70 bp | 100.0% | 1 | 2.2M | 58.2% |
| 79 | US | USUSDT (usdm-futures) | usdm-futures | 88.4% | 0.168 | BLUAIUSDT | 209.81 bp | 100.0% | 0 | 29.4M | 306.6% |
| 80 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 88.4% | 0.121 | THEUSDT | 111.94 bp | 100.0% | 1 | 21.5M | 147.9% |
| 81 | PORTO | PORTOUSDT (spot) | spot | 88.7% | 0.133 | TLMUSDT | 111.13 bp | 100.0% | 2 | 526.7K | 244.8% |
| 82 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 88.3% | 0.142 | TRADOORUSDT | 101.25 bp | 100.0% | 1 | 7.1M | 134.5% |
| 83 | PIVX | PIVXUSDT (spot) | spot | 87.8% | 0.257 | BLESSUSDT | 100.18 bp | 100.0% | 2 | 298.5K | 144.6% |
| 84 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 88.0% | 0.196 | BEATUSDT | 99.77 bp | 100.0% | 2 | 624.7K | 176.7% |
| 85 | ERA | ERAUSDT (spot) | spot, usdm-futures | 87.9% | 0.181 | BTCUSDT | 98.53 bp | 100.0% | 3 | 486.2K | 233.4% |
| 86 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 87.9% | 0.205 | BTCUSDT | 72.67 bp | 100.0% | 4 | 618.9K | 96.8% |
| 87 | HEI | HEIUSDT (spot) | spot, usdm-futures | 87.1% | 0.161 | AERGOUSDT | 127.92 bp | 100.0% | 1 | 1.9M | 160.2% |
| 88 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 87.2% | 0.146 | ZEREBROUSDT | 105.87 bp | 100.0% | 2 | 2.2M | 158.8% |
| 89 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 86.2% | 0.212 | DODOUSDT | 116.04 bp | 100.0% | 2 | 1.7M | 230.8% |
| 90 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 86.5% | 0.174 | EVAAUSDT | 74.83 bp | 100.0% | 2 | 858.9K | 117.5% |
| 91 | YB | YBUSDT (spot) | spot, usdm-futures | 85.9% | 0.238 | BLUAIUSDT | 75.85 bp | 100.0% | 2 | 328.2K | 116.8% |
| 92 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 85.9% | 0.154 | THEUSDT | 125.42 bp | 100.0% | 0 | 13.9M | 157.7% |
| 93 | KITE | KITEUSDT (spot) | spot, usdm-futures | 85.5% | 0.154 | BUSDT | 88.37 bp | 100.0% | 1 | 11.5M | 112.0% |
| 94 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 85.3% | 0.143 | EVAAUSDT | 79.88 bp | 100.0% | 1 | 2.8M | 127.0% |
| 95 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 84.9% | 0.188 | BTCUSDT | 90.00 bp | 100.0% | 1 | 44.4M | 123.8% |
| 96 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 84.8% | 0.149 | ACEUSDT | 85.69 bp | 100.0% | 1 | 2.5M | 122.8% |
| 97 | OPN | OPNUSDT (spot) | spot, usdm-futures | 85.0% | 0.158 | USUSDT | 79.52 bp | 100.0% | 3 | 15.1M | 112.2% |
| 98 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 85.0% | 0.207 | BTCUSDT | 71.26 bp | 100.0% | 2 | 768.7K | 102.2% |
| 99 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 85.0% | 0.284 | BTCUSDT | 70.46 bp | 100.0% | 1 | 1.3M | 89.7% |
| 100 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 85.0% | 0.281 | THEUSDT | 54.48 bp | 100.0% | 1 | 832.6K | 81.4% |
| 101 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 83.9% | 0.194 | BTCUSDT | 145.29 bp | 100.0% | 0 | 12.6M | 202.0% |
| 102 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 83.9% | 0.189 | PROMUSDT | 82.26 bp | 100.0% | 2 | 1M | 145.3% |
| 103 | A | AUSDT (spot) | spot, usdm-futures | 84.0% | 0.219 | BTCUSDT | 58.10 bp | 100.0% | 2 | 385.1K | 78.4% |
| 104 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 84.0% | 0.173 | MSFTBUSDT | 52.61 bp | 100.0% | 2 | 391.6K | 76.2% |
| 105 | U | UUSDT (spot) | spot | 83.8% | 0.149 | B2USDT | 0.54 bp | 100.0% | 7 | 14.9M | 0.7% |
| 106 | BR | BRUSDT (usdm-futures) | usdm-futures | 83.0% | 0.172 | QUICKUSDT | 78.78 bp | 100.0% | 1 | 945.2K | 109.9% |
| 107 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 82.9% | 0.181 | QIUSDT | 78.36 bp | 100.0% | 1 | 2.2M | 107.6% |
| 108 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 82.7% | 0.145 | YBUSDT | 67.31 bp | 100.0% | 2 | 1.2M | 91.1% |
| 109 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 82.3% | 0.263 | BTCUSDT | 62.62 bp | 100.0% | 2 | 329.9K | 84.8% |
| 110 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 82.3% | 0.285 | BTCUSDT | 61.93 bp | 100.0% | 1 | 1.5M | 91.1% |
| 111 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 82.6% | 0.212 | OPNUSDT | 61.75 bp | 100.0% | 1 | 452K | 78.1% |
| 112 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 81.8% | 0.191 | PYRUSDT | 122.20 bp | 100.0% | 1 | 8.1M | 183.8% |
| 113 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 81.7% | 0.185 | BULLAUSDT | 81.90 bp | 100.0% | 0 | 1.5M | 106.8% |
| 114 | HOT | HOTUSDT (spot) | spot, usdm-futures | 81.5% | 0.286 | BTCUSDT | 70.57 bp | 100.0% | 3 | 256K | 108.1% |
| 115 | WIN | WINUSDT (spot) | spot | 81.8% | 0.330 | BTCUSDT | 38.63 bp | 100.0% | 2 | 97.9K | 51.7% |
| 116 | STAR | STARUSDT (usdm-futures) | usdm-futures | 81.1% | 0.153 | TREEUSDT | 138.31 bp | 100.0% | 0 | 2.5M | 213.3% |
| 117 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 81.1% | 0.193 | BTCUSDT | 30.74 bp | 100.0% | 3 | 2.5M | 38.9% |
| 118 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 80.3% | 0.238 | ARXUSDT | 70.16 bp | 100.0% | 1 | 2.9M | 95.0% |
| 119 | AUDIO | AUDIOUSDT (spot) | spot | 80.3% | 0.205 | AMPUSDT | 39.47 bp | 100.0% | 2 | 373.7K | 75.9% |
| 120 | KNC | KNCUSDT (spot) | spot, usdm-futures | 80.1% | 0.220 | BTCUSDT | 36.53 bp | 100.0% | 3 | 65.5K | 76.0% |
| 121 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 79.9% | 0.266 | TAKEUSDT | 23.19 bp | 100.0% | 10 | 189.7K | 37.6% |
| 122 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 79.5% | 0.201 | MSFTBUSDT | 25.05 bp | 100.0% | 1 | 260.5K | 44.2% |
| 123 | VANA | VANAUSDT (spot) | spot, usdm-futures | 79.4% | 0.260 | EGLDUSDT | 59.55 bp | 100.0% | 2 | 1.1M | 81.0% |
| 124 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 79.3% | 0.171 | AKEUSDT | 104.44 bp | 100.0% | 3 | 3.4M | 147.0% |
| 125 | AI | AIUSDT (spot) | spot | 79.1% | 0.155 | ENSOUSDT | 61.52 bp | 100.0% | 5 | 342.4K | 110.1% |
| 126 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 78.7% | 0.229 | SKYAIUSDT | 129.22 bp | 100.0% | 1 | 5.9M | 185.4% |
| 127 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 78.3% | 0.139 | ONUSDT | 121.96 bp | 100.0% | 1 | 2.9M | 152.4% |
| 128 | BAS | BASUSDT (usdm-futures) | usdm-futures | 78.6% | 0.184 | PIVXUSDT | 119.64 bp | 100.0% | 1 | 3.2M | 170.9% |
| 129 | TUT | TUTUSDT (spot) | spot, usdm-futures | 78.5% | 0.191 | HOTUSDT | 86.29 bp | 100.0% | 1 | 471K | 112.6% |
| 130 | 4 | 4USDT (usdm-futures) | usdm-futures | 78.1% | 0.183 | ANTHROPICUSDT | 57.06 bp | 100.0% | 1 | 1.8M | 74.0% |
| 131 | Q | QUSDT (usdm-futures) | usdm-futures | 77.8% | 0.158 | OUSDT | 80.55 bp | 100.0% | 1 | 1.7M | 115.1% |
| 132 | FTT | FTTUSDT (spot) | spot | 78.0% | 0.343 | BTCUSDT | 56.24 bp | 100.0% | 2 | 190K | 84.5% |
| 133 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 77.5% | 0.285 | BANKUSDT | 223.19 bp | 100.0% | 1 | 22.3M | 452.5% |
| 134 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 77.8% | 0.178 | STARUSDT | 49.05 bp | 100.0% | 1 | 1.7M | 87.8% |
| 135 | SXT | SXTUSDT (spot) | spot, usdm-futures | 77.0% | 0.176 | BSBUSDT | 130.75 bp | 100.0% | 2 | 4.1M | 200.9% |
| 136 | H | HUSDT (usdm-futures) | usdm-futures | 76.9% | 0.189 | LUMIAUSDT | 85.76 bp | 100.0% | 1 | 6.7M | 121.5% |
| 137 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 76.3% | 0.249 | GLMRUSDT | 104.48 bp | 100.0% | 0 | 2.7M | 211.0% |
| 138 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 76.5% | 0.284 | BTCUSDT | 65.92 bp | 100.0% | 1 | 2.1M | 85.7% |
| 139 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 76.2% | 0.158 | BTWUSDT | 88.09 bp | 100.0% | 4 | 1.3M | 181.2% |
| 140 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 76.1% | 0.252 | BTCUSDT | 75.13 bp | 100.0% | 1 | 1.1M | 102.1% |
| 141 | ASTS | ASTSUSDT (usdm-futures) | usdm-futures | 75.9% | 0.220 | BTCUSDT | 66.57 bp | 100.0% | 1 | 2.5M | 120.2% |
| 142 | MAV | MAVUSDT (spot) | spot, usdm-futures | 76.0% | 0.325 | BTCUSDT | 54.09 bp | 100.0% | 2 | 179.9K | 90.0% |
| 143 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 75.4% | 0.162 | ALCHUSDT | 180.81 bp | 100.0% | 0 | 9.2M | 245.5% |
| 144 | BEL | BELUSDT (spot) | spot, usdm-futures | 75.2% | 0.161 | HEIUSDT | 79.04 bp | 100.0% | 1 | 741.2K | 98.8% |
| 145 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 75.4% | 0.205 | SLXUSDT | 61.19 bp | 100.0% | 2 | 1.4M | 82.5% |
| 146 | RIVN | RIVNUSDT (usdm-futures) | usdm-futures | 75.5% | 0.251 | QIUSDT | 43.33 bp | 100.0% | 3 | 393.4K | 71.4% |
| 147 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 74.6% | 0.201 | PROMUSDT | 66.84 bp | 100.0% | 1 | 2M | 91.5% |
| 148 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 74.5% | 0.278 | BLESSUSDT | 85.32 bp | 100.0% | 1 | 1M | 136.5% |
| 149 | T | TUSDT (spot) | spot, usdm-futures | 74.2% | 0.307 | ANKRUSDT | 107.60 bp | 100.0% | 4 | 1.7M | 198.9% |
| 150 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 74.3% | 0.216 | ERAUSDT | 60.68 bp | 100.0% | 1 | 1.1M | 84.2% |
| 151 | RE | REUSDT (spot) | spot, usdm-futures | 73.7% | 0.234 | SAPIENUSDT | 99.00 bp | 100.0% | 1 | 14.1M | 126.4% |
| 152 | ID | IDUSDT (spot) | spot, usdm-futures | 73.8% | 0.244 | CHIPUSDT | 68.92 bp | 100.0% | 2 | 702.3K | 89.4% |
| 153 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 73.1% | 0.263 | BLESSUSDT | 70.84 bp | 100.0% | 3 | 2.2M | 95.0% |
| 154 | 2Z | 2ZUSDT (spot) | spot, usdm-futures | 73.4% | 0.317 | BTCUSDT | 50.48 bp | 100.0% | 1 | 200.1K | 65.9% |
| 155 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 73.0% | 0.291 | BTCUSDT | 49.26 bp | 100.0% | 5 | 664K | 64.9% |
| 156 | SUN | SUNUSDT (spot) | spot, usdm-futures | 73.2% | 0.196 | THEUSDT | 17.68 bp | 100.0% | 3 | 644.6K | 24.6% |
| 157 | TA | TAUSDT (usdm-futures) | usdm-futures | 72.8% | 0.247 | KNCUSDT | 46.74 bp | 100.0% | 1 | 1.1M | 66.2% |
| 158 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 71.9% | 0.206 | BASUSDT | 187.25 bp | 100.0% | 1 | 23.9M | 259.3% |
| 159 | JTO | JTOUSDT (spot) | spot, usdm-futures | 71.9% | 0.203 | PIVXUSDT | 84.28 bp | 100.0% | 1 | 3.1M | 115.5% |
| 160 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 71.5% | 0.247 | BTCUSDT | 72.86 bp | 100.0% | 1 | 867.4K | 98.3% |
| 161 | GPS | GPSUSDT (spot) | spot, usdm-futures | 71.4% | 0.185 | BTCUSDT | 64.48 bp | 100.0% | 2 | 375.2K | 90.6% |
| 162 | NMR | NMRUSDT (spot) | spot, usdm-futures | 71.5% | 0.221 | BTCUSDT | 45.16 bp | 100.0% | 2 | 220.2K | 58.8% |
| 163 | MBL | MBLUSDT (spot) | spot | 71.3% | 0.248 | BTCUSDT | 39.92 bp | 100.0% | 3 | 450.3K | 51.7% |
| 164 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 70.5% | 0.212 | BTCUSDT | 70.01 bp | 100.0% | 2 | 1.7M | 104.6% |
| 165 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 70.8% | 0.266 | MUSDT | 67.86 bp | 100.0% | 2 | 1.5M | 91.9% |
| 166 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 70.2% | 0.268 | BTCUSDT | 63.46 bp | 100.0% | 2 | 1M | 97.3% |
| 167 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 69.9% | 0.373 | BTCUSDT | 65.41 bp | 100.0% | 1 | 3.1M | 84.5% |
| 168 | ZRO | ZROUSDT (spot) | spot, usdm-futures | 69.8% | 0.280 | BTCUSDT | 63.84 bp | 100.0% | 1 | 1.9M | 80.4% |
| 169 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 69.6% | 0.253 | KNCUSDT | 63.35 bp | 100.0% | 1 | 416.6K | 82.0% |
| 170 | GNS | GNSUSDT (spot) | spot | 69.7% | 0.252 | ACEUSDT | 27.40 bp | 100.0% | 5 | 50.7K | 41.7% |
| 171 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 68.9% | 0.248 | BTCUSDT | 86.04 bp | 100.0% | 1 | 1.5M | 115.6% |
| 172 | PLUME | PLUMEUSDT (spot) | spot, usdm-futures | 69.0% | 0.332 | BTCUSDT | 66.90 bp | 100.0% | 1 | 1.1M | 84.1% |
| 173 | RARE | RAREUSDT (spot) | spot, usdm-futures | 68.9% | 0.320 | PYRUSDT | 54.70 bp | 100.0% | 12 | 196.9K | 100.6% |
| 174 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 68.4% | 0.192 | ANTHROPICUSDT | 48.05 bp | 100.0% | 1 | 587K | 66.7% |
| 175 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 68.7% | 0.209 | DISUSDT | 18.42 bp | 100.0% | 1 | 17M | 26.2% |
| 176 | CC | CCUSDT (usdm-futures) | usdm-futures | 68.2% | 0.160 | DATAIPUSDT | 44.93 bp | 100.0% | 1 | 3.7M | 58.6% |
| 177 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 67.9% | 0.236 | TRIAUSDT | 54.73 bp | 100.0% | 2 | 1M | 77.9% |
| 178 | MITO | MITOUSDT (spot) | spot, usdm-futures | 67.2% | 0.289 | THEUSDT | 74.07 bp | 100.0% | 2 | 634.3K | 107.3% |
| 179 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 67.5% | 0.307 | BTCUSDT | 52.04 bp | 100.0% | 4 | 1.9M | 80.3% |
| 180 | FF | FFUSDT (spot) | spot, usdm-futures | 67.3% | 0.236 | AKEUSDT | 48.43 bp | 100.0% | 1 | 1.2M | 77.6% |
| 181 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 67.2% | 0.353 | BTCUSDT | 47.88 bp | 100.0% | 2 | 573.6K | 65.0% |
| 182 | GRAM | GRAMUSDT (spot) | spot, usdm-futures | 67.3% | 0.335 | BTCUSDT | 47.18 bp | 100.0% | 2 | 7.4M | 66.4% |
| 183 | DCR | DCRUSDT (spot) | spot | 66.7% | 0.281 | LUMIAUSDT | 83.00 bp | 100.0% | 2 | 250.6K | 164.0% |
| 184 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 66.6% | 0.289 | TOSHIUSDT | 52.85 bp | 100.0% | 1 | 150.7K | 72.3% |
| 185 | ESP | ESPUSDT (spot) | spot, usdm-futures | 66.5% | 0.193 | PYRUSDT | 49.78 bp | 100.0% | 1 | 382K | 66.6% |
| 186 | MASK | MASKUSDT (spot) | spot, usdm-futures | 65.9% | 0.367 | BTCUSDT | 38.75 bp | 100.0% | 5 | 195.3K | 51.1% |
| 187 | BX | BXUSDT (usdm-futures) | usdm-futures | 66.0% | 0.300 | RIVNUSDT | 33.19 bp | 100.0% | 1 | 804.9K | 47.8% |
| 188 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 65.6% | 0.216 | AIOUSDT | 75.23 bp | 100.0% | 2 | 558.8K | 102.8% |
| 189 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 65.4% | 0.273 | BTCUSDT | 47.59 bp | 100.0% | 2 | 167.4K | 64.8% |
| 190 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 65.2% | 0.180 | RIFUSDT | 22.97 bp | 100.0% | 2 | 212.8K | 33.8% |
| 191 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 64.5% | 0.270 | EDGEUSDT | 222.99 bp | 100.0% | 1 | 9.7M | 445.2% |
| 192 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 64.7% | 0.264 | ACEUSDT | 128.08 bp | 100.0% | 1 | 1.6M | 199.4% |
| 193 | SENT | SENTUSDT (spot) | spot, usdm-futures | 64.6% | 0.253 | FIGHTUSDT | 85.77 bp | 100.0% | 1 | 1.8M | 131.4% |
| 194 | APR | APRUSDT (usdm-futures) | usdm-futures | 64.4% | 0.286 | AIOTUSDT | 75.12 bp | 100.0% | 2 | 3.3M | 105.2% |
| 195 | BANANA | BANANAUSDT (spot) | spot, usdm-futures | 64.5% | 0.312 | BTCUSDT | 57.21 bp | 100.0% | 1 | 378.7K | 75.2% |
| 196 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 63.5% | 0.259 | DGBUSDT | 87.82 bp | 100.0% | 1 | 3.7M | 146.6% |
| 197 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 63.6% | 0.282 | BTCUSDT | 56.75 bp | 100.0% | 2 | 4.5M | 84.2% |
| 198 | BONK | BONKUSDT (spot) | spot, usdm-futures | 63.1% | 0.422 | BTCUSDT | 73.58 bp | 100.0% | 3 | 3.7M | 96.5% |
| 199 | STRAX | STRAXUSDT (spot) | spot | 63.4% | 0.261 | MAGMAUSDT | 53.16 bp | 100.0% | 2 | 355.8K | 116.5% |
| 200 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 62.6% | 0.207 | ROBOUSDT | 77.83 bp | 100.0% | 1 | 3M | 98.9% |
| 201 | KAVA | KAVAUSDT (spot) | spot, usdm-futures | 62.8% | 0.297 | KNCUSDT | 23.88 bp | 100.0% | 1 | 356.5K | 37.6% |
| 202 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 62.4% | 0.206 | REUSDT | 66.73 bp | 100.0% | 2 | 841.4K | 87.9% |
| 203 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 61.9% | 0.231 | SKYAIUSDT | 110.04 bp | 100.0% | 1 | 2.6M | 146.5% |
| 204 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 61.4% | 0.221 | GNOUSDT | 79.97 bp | 100.0% | 1 | 961.2K | 108.1% |
| 205 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 61.5% | 0.179 | TAKEUSDT | 79.83 bp | 100.0% | 1 | 1.3M | 99.8% |
| 206 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 61.4% | 0.379 | BTCUSDT | 37.82 bp | 100.0% | 0 | 23M | 45.8% |
| 207 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 60.7% | 0.381 | TLMUSDT | 177.13 bp | 100.0% | 1 | 5.3M | 245.1% |
| 208 | UB | UBUSDT (usdm-futures) | usdm-futures | 60.8% | 0.214 | USUSDT | 153.31 bp | 100.0% | 1 | 14.5M | 193.9% |
| 209 | WET | WETUSDT (usdm-futures) | usdm-futures | 60.7% | 0.267 | BTCUSDT | 71.49 bp | 100.0% | 1 | 1.5M | 94.0% |
| 210 | STORJ | STORJUSDT (spot) | spot, usdm-futures | 60.3% | 0.386 | BTCUSDT | 44.59 bp | 100.0% | 3 | 181.5K | 74.5% |
| 211 | TFUEL | TFUELUSDT (spot) | spot | 60.5% | 0.323 | BTCUSDT | 40.19 bp | 100.0% | 2 | 98.2K | 50.1% |
| 212 | C | CUSDT (spot) | spot, usdm-futures | 59.9% | 0.320 | BTCUSDT | 59.94 bp | 100.0% | 3 | 294.2K | 85.8% |
| 213 | SC | SCUSDT (spot) | spot | 59.9% | 0.446 | BTCUSDT | 32.88 bp | 100.0% | 5 | 117.2K | 43.6% |
| 214 | SKY | SKYUSDT (spot) | spot, usdm-futures | 58.9% | 0.290 | BTCUSDT | 64.93 bp | 100.0% | 1 | 1.2M | 80.0% |
| 215 | PYTH | PYTHUSDT (spot) | spot, usdm-futures | 58.8% | 0.260 | BTCUSDT | 71.45 bp | 100.0% | 1 | 1.9M | 85.6% |
| 216 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 58.7% | 0.287 | BTCUSDT | 47.14 bp | 100.0% | 4 | 488.5K | 61.1% |
| 217 | GME | GMEUSDT (usdm-futures) | usdm-futures | 58.9% | 0.290 | BTCUSDT | 17.93 bp | 100.0% | 2 | 218.3K | 25.8% |
| 218 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 58.0% | 0.241 | PIPPINUSDT | 102.24 bp | 100.0% | 2 | 7.8M | 138.4% |
| 219 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 58.0% | 0.255 | PORTOUSDT | 62.06 bp | 100.0% | 1 | 1.7M | 90.3% |
| 220 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 57.8% | 0.447 | BLESSUSDT | 92.02 bp | 100.0% | 1 | 691.1K | 191.1% |
| 221 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 57.8% | 0.255 | AERGOUSDT | 66.12 bp | 100.0% | 1 | 3.6M | 103.4% |

## Diagnostics

- Basis size selected: 221
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.056
- Maximum pairwise absolute correlation: 0.447
- Mean whole-market projection R²: 84.2%
- Median whole-market projection R²: 80.0%
- 10th-percentile whole-market projection R²: 70.9%
- Minimum whole-market projection R²: 66.5%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.4% | 10.8% | 0.5% | 0.0% |
| 5 | 17.3% | 12.6% | 1.8% | 0.1% |
| 10 | 19.6% | 14.6% | 3.5% | 0.7% |
| 15 | 21.8% | 16.3% | 4.8% | 1.4% |
| 20 | 23.7% | 17.9% | 6.1% | 3.1% |
| 25 | 25.7% | 19.6% | 7.8% | 3.9% |
| 30 | 28.6% | 22.9% | 9.6% | 5.7% |
| 35 | 30.5% | 24.6% | 11.7% | 7.5% |
| 40 | 32.5% | 26.3% | 13.5% | 8.6% |
| 45 | 34.8% | 29.0% | 15.2% | 10.2% |
| 50 | 36.8% | 30.4% | 16.8% | 12.9% |
| 55 | 38.7% | 32.9% | 18.5% | 13.6% |
| 60 | 40.7% | 34.4% | 20.1% | 15.5% |
| 65 | 42.4% | 36.0% | 21.8% | 16.4% |
| 70 | 44.2% | 37.9% | 23.4% | 18.1% |
| 75 | 46.1% | 39.3% | 25.2% | 20.1% |
| 80 | 47.8% | 41.1% | 26.7% | 21.3% |
| 85 | 49.7% | 43.1% | 28.5% | 22.8% |
| 90 | 51.3% | 44.4% | 30.0% | 25.4% |
| 95 | 53.1% | 46.1% | 31.9% | 27.4% |
| 100 | 54.8% | 47.7% | 33.8% | 29.2% |
| 105 | 56.4% | 49.2% | 35.5% | 31.2% |
| 110 | 57.9% | 50.8% | 37.2% | 31.8% |
| 115 | 59.4% | 52.3% | 39.1% | 34.1% |
| 120 | 61.0% | 54.1% | 40.8% | 35.9% |
| 125 | 62.6% | 56.1% | 42.3% | 37.6% |
| 130 | 63.9% | 57.3% | 43.9% | 39.0% |
| 135 | 65.2% | 58.7% | 45.6% | 40.8% |
| 140 | 66.5% | 60.4% | 47.2% | 42.2% |
| 145 | 67.9% | 62.2% | 48.8% | 43.0% |
| 150 | 69.1% | 63.5% | 50.4% | 45.4% |
| 155 | 70.5% | 64.7% | 51.9% | 46.4% |
| 160 | 71.7% | 66.0% | 53.5% | 48.8% |
| 165 | 72.8% | 66.9% | 54.6% | 50.3% |
| 170 | 74.0% | 68.4% | 56.4% | 52.3% |
| 175 | 75.2% | 69.5% | 57.8% | 53.4% |
| 180 | 76.3% | 70.9% | 59.2% | 54.7% |
| 185 | 77.3% | 72.1% | 60.7% | 56.3% |
| 190 | 78.4% | 73.5% | 62.3% | 58.0% |
| 195 | 79.4% | 74.3% | 63.6% | 59.3% |
| 200 | 80.4% | 75.8% | 65.0% | 60.5% |
| 205 | 81.3% | 76.5% | 66.5% | 62.3% |
| 210 | 82.3% | 77.9% | 68.1% | 63.4% |
| 215 | 83.1% | 78.7% | 69.0% | 65.3% |
| 220 | 84.0% | 79.8% | 70.7% | 66.4% |
| 221 | 84.2% | 80.0% | 70.9% | 66.5% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | EVAA | BANK | AKE | TLM | TRIA | DODO | ON | EDGE | MAGMA | PYR | BTTC | ALLO | BLUAI | XEC | ATM | XNO | ALCH | AERGO | QUICK | VELVET | AIN | EPIC | AWE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.072 | 0.033 | 0.027 | -0.047 | 0.083 | 0.028 | 0.007 | 0.032 | 0.022 | 0.076 | 0.067 | 0.073 | -0.025 | 0.052 | -0.059 | -0.047 | -0.020 | 0.010 | 0.087 | 0.008 | 0.033 | 0.056 | 0.113 |
| EVAA | 0.072 | 1.000 | 0.021 | 0.031 | -0.004 | 0.024 | 0.005 | 0.064 | 0.042 | -0.068 | 0.004 | -0.008 | -0.005 | 0.031 | 0.043 | 0.021 | 0.029 | 0.011 | -0.004 | 0.007 | 0.023 | -0.013 | -0.031 | -0.048 |
| BANK | 0.033 | 0.021 | 1.000 | -0.028 | 0.080 | 0.024 | -0.014 | 0.027 | 0.005 | -0.031 | -0.032 | 0.007 | -0.006 | -0.009 | 0.065 | -0.013 | 0.095 | 0.075 | 0.088 | 0.051 | 0.089 | -0.081 | -0.093 | 0.013 |
| AKE | 0.027 | 0.031 | -0.028 | 1.000 | -0.002 | -0.019 | -0.019 | -0.039 | -0.005 | 0.002 | 0.006 | -0.031 | -0.022 | 0.070 | 0.053 | 0.023 | -0.016 | -0.073 | 0.033 | 0.013 | -0.038 | -0.016 | 0.049 | 0.010 |
| TLM | -0.047 | -0.004 | 0.080 | -0.002 | 1.000 | -0.021 | 0.003 | 0.067 | 0.030 | -0.042 | 0.043 | -0.054 | 0.015 | 0.007 | -0.047 | 0.014 | 0.001 | -0.033 | -0.033 | 0.013 | 0.093 | 0.001 | 0.022 | 0.039 |
| TRIA | 0.083 | 0.024 | 0.024 | -0.019 | -0.021 | 1.000 | 0.041 | -0.022 | -0.028 | 0.013 | 0.033 | 0.007 | -0.030 | -0.012 | -0.013 | 0.014 | -0.038 | -0.011 | 0.029 | 0.031 | 0.063 | -0.012 | -0.005 | -0.015 |
| DODO | 0.028 | 0.005 | -0.014 | -0.019 | 0.003 | 0.041 | 1.000 | 0.016 | 0.020 | 0.028 | 0.011 | -0.018 | -0.007 | -0.017 | -0.029 | 0.012 | 0.005 | 0.050 | 0.041 | -0.021 | -0.007 | -0.050 | 0.064 | -0.056 |
| ON | 0.007 | 0.064 | 0.027 | -0.039 | 0.067 | -0.022 | 0.016 | 1.000 | 0.014 | 0.008 | -0.023 | -0.017 | -0.022 | -0.030 | 0.011 | 0.033 | 0.103 | -0.006 | 0.060 | 0.040 | 0.055 | 0.076 | 0.076 | 0.029 |
| EDGE | 0.032 | 0.042 | 0.005 | -0.005 | 0.030 | -0.028 | 0.020 | 0.014 | 1.000 | 0.081 | 0.047 | 0.016 | 0.027 | -0.017 | -0.022 | -0.027 | 0.036 | 0.016 | -0.037 | 0.032 | 0.059 | 0.016 | 0.001 | -0.016 |
| MAGMA | 0.022 | -0.068 | -0.031 | 0.002 | -0.042 | 0.013 | 0.028 | 0.008 | 0.081 | 1.000 | 0.021 | -0.045 | 0.001 | 0.056 | -0.024 | -0.012 | -0.018 | 0.016 | 0.013 | -0.075 | 0.088 | 0.047 | 0.005 | -0.028 |
| PYR | 0.076 | 0.004 | -0.032 | 0.006 | 0.043 | 0.033 | 0.011 | -0.023 | 0.047 | 0.021 | 1.000 | 0.005 | -0.011 | 0.038 | -0.008 | 0.031 | 0.016 | 0.022 | 0.042 | 0.025 | -0.001 | 0.022 | -0.011 | 0.005 |
| BTTC | 0.067 | -0.008 | 0.007 | -0.031 | -0.054 | 0.007 | -0.018 | -0.017 | 0.016 | -0.045 | 0.005 | 1.000 | -0.007 | -0.010 | -0.031 | 0.030 | -0.016 | 0.104 | -0.025 | 0.012 | -0.040 | 0.010 | 0.017 | -0.028 |
| ALLO | 0.073 | -0.005 | -0.006 | -0.022 | 0.015 | -0.030 | -0.007 | -0.022 | 0.027 | 0.001 | -0.011 | -0.007 | 1.000 | -0.019 | 0.003 | -0.046 | 0.019 | 0.002 | 0.022 | -0.013 | 0.012 | 0.049 | 0.009 | 0.033 |
| BLUAI | -0.025 | 0.031 | -0.009 | 0.070 | 0.007 | -0.012 | -0.017 | -0.030 | -0.017 | 0.056 | 0.038 | -0.010 | -0.019 | 1.000 | 0.066 | 0.011 | 0.038 | -0.012 | 0.044 | 0.018 | 0.015 | 0.060 | -0.045 | -0.023 |
| XEC | 0.052 | 0.043 | 0.065 | 0.053 | -0.047 | -0.013 | -0.029 | 0.011 | -0.022 | -0.024 | -0.008 | -0.031 | 0.003 | 0.066 | 1.000 | 0.049 | 0.022 | 0.008 | -0.007 | 0.074 | -0.002 | 0.067 | -0.055 | -0.065 |
| ATM | -0.059 | 0.021 | -0.013 | 0.023 | 0.014 | 0.014 | 0.012 | 0.033 | -0.027 | -0.012 | 0.031 | 0.030 | -0.046 | 0.011 | 0.049 | 1.000 | -0.017 | -0.015 | -0.044 | 0.063 | 0.036 | -0.059 | 0.003 | 0.041 |
| XNO | -0.047 | 0.029 | 0.095 | -0.016 | 0.001 | -0.038 | 0.005 | 0.103 | 0.036 | -0.018 | 0.016 | -0.016 | 0.019 | 0.038 | 0.022 | -0.017 | 1.000 | -0.010 | -0.058 | 0.026 | -0.006 | -0.004 | -0.036 | -0.013 |
| ALCH | -0.020 | 0.011 | 0.075 | -0.073 | -0.033 | -0.011 | 0.050 | -0.006 | 0.016 | 0.016 | 0.022 | 0.104 | 0.002 | -0.012 | 0.008 | -0.015 | -0.010 | 1.000 | -0.020 | -0.034 | 0.016 | 0.054 | 0.045 | 0.009 |
| AERGO | 0.010 | -0.004 | 0.088 | 0.033 | -0.033 | 0.029 | 0.041 | 0.060 | -0.037 | 0.013 | 0.042 | -0.025 | 0.022 | 0.044 | -0.007 | -0.044 | -0.058 | -0.020 | 1.000 | 0.053 | -0.007 | -0.044 | -0.035 | 0.022 |
| QUICK | 0.087 | 0.007 | 0.051 | 0.013 | 0.013 | 0.031 | -0.021 | 0.040 | 0.032 | -0.075 | 0.025 | 0.012 | -0.013 | 0.018 | 0.074 | 0.063 | 0.026 | -0.034 | 0.053 | 1.000 | 0.010 | -0.019 | -0.028 | -0.005 |
| VELVET | 0.008 | 0.023 | 0.089 | -0.038 | 0.093 | 0.063 | -0.007 | 0.055 | 0.059 | 0.088 | -0.001 | -0.040 | 0.012 | 0.015 | -0.002 | 0.036 | -0.006 | 0.016 | -0.007 | 0.010 | 1.000 | -0.043 | 0.073 | -0.025 |
| AIN | 0.033 | -0.013 | -0.081 | -0.016 | 0.001 | -0.012 | -0.050 | 0.076 | 0.016 | 0.047 | 0.022 | 0.010 | 0.049 | 0.060 | 0.067 | -0.059 | -0.004 | 0.054 | -0.044 | -0.019 | -0.043 | 1.000 | 0.021 | 0.058 |
| EPIC | 0.056 | -0.031 | -0.093 | 0.049 | 0.022 | -0.005 | 0.064 | 0.076 | 0.001 | 0.005 | -0.011 | 0.017 | 0.009 | -0.045 | -0.055 | 0.003 | -0.036 | 0.045 | -0.035 | -0.028 | 0.073 | 0.021 | 1.000 | 0.018 |
| AWE | 0.113 | -0.048 | 0.013 | 0.010 | 0.039 | -0.015 | -0.056 | 0.029 | -0.016 | -0.028 | 0.005 | -0.028 | 0.033 | -0.023 | -0.065 | 0.041 | -0.013 | 0.009 | 0.022 | -0.005 | -0.025 | 0.058 | 0.018 | 1.000 |

The complete 221 × 221 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| REQ | REQUSDT | 66.5% | 57.9% | BTCUSDT | 0.348 |
| PLAY | PLAYUSDT | 66.9% | 57.5% | AMPUSDT | 0.297 |
| HFT | HFTUSDT | 67.1% | 57.4% | BTCUSDT | 0.341 |
| BABY | BABYUSDT | 67.3% | 57.2% | BTCUSDT | 0.440 |
| TAC | TACUSDT | 67.4% | 57.1% | TRIAUSDT | 0.241 |
| NAORIS | NAORISUSDT | 67.7% | 56.8% | ERAUSDT | -0.215 |
| GUA | GUAUSDT | 67.8% | 56.7% | SKYAIUSDT | 0.259 |
| ZKP | ZKPUSDT | 67.8% | 56.7% | BTCUSDT | 0.293 |
| SONY | SONYUSDT | 67.9% | 56.7% | GMEUSDT | 0.235 |
| STRC | STRCUSDT | 67.9% | 56.7% | BTCUSDT | 0.267 |
| INJ | INJUSDT | 67.9% | 56.6% | BTCUSDT | 0.419 |
| LDO | LDOUSDT | 68.0% | 56.6% | BTCUSDT | 0.339 |
| LA | LAUSDT | 68.1% | 56.5% | ERAUSDT | 0.420 |
| ACM | ACMUSDT | 68.1% | 56.5% | BARUSDT | 0.379 |
| COOKIE | COOKIEUSDT | 68.1% | 56.4% | BTCUSDT | 0.301 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

