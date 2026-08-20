# Binance portfolio basis

Generated 2026-07-23T19:17:26.889Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-22T18:00Z through 2026-07-22T23:59Z
- Sampling: exactly 360 1m log returns (0.3 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 5.0% of the maximum unexplained variance, select the largest mean absolute 1m return
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 713 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 2.48 bp | 100.0% | 0 | 133.9M | 24.9% |
| 2 | XNO | XNOUSDT (spot) | spot | 99.8% | 0.058 | BTCUSDT | 71.23 bp | 100.0% | 0.1 | 849.2K | 825.7% |
| 3 | BTTC | BTTCUSDT (spot) | spot | 99.7% | 0.059 | BTCUSDT | 64.65 bp | 100.0% | 0.7 | 19.3K | 1113.2% |
| 4 | RIF | RIFUSDT (spot) | spot, usdm-futures | 99.7% | 0.075 | XNOUSDT | 55.76 bp | 100.0% | 0.1 | 1.4M | 624.4% |
| 5 | B2 | B2USDT (usdm-futures) | usdm-futures | 99.8% | 0.051 | BTCUSDT | 49.97 bp | 100.0% | 0 | 13M | 615.2% |
| 6 | BANK | BANKUSDT (spot) | spot, usdm-futures | 98.7% | 0.138 | BTCUSDT | 46.41 bp | 100.0% | 0 | 18.5M | 496.0% |
| 7 | PYR | PYRUSDT (spot) | spot | 99.1% | 0.091 | B2USDT | 39.29 bp | 100.0% | 1 | 184.6K | 495.6% |
| 8 | ONE | ONEUSDT (spot) | spot, usdm-futures | 99.4% | 0.070 | BANKUSDT | 34.95 bp | 100.0% | 0.2 | 223.5K | 382.6% |
| 9 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 99.2% | 0.077 | BTTCUSDT | 33.19 bp | 100.0% | 0 | 4M | 327.3% |
| 10 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.4% | 0.095 | BTTCUSDT | 31.24 bp | 100.0% | 0 | 26.5M | 301.7% |
| 11 | ON | ONUSDT (usdm-futures) | usdm-futures | 98.4% | 0.105 | ONEUSDT | 31.09 bp | 100.0% | 0 | 13.6M | 295.6% |
| 12 | ERA | ERAUSDT (spot) | spot, usdm-futures | 97.3% | 0.125 | ONUSDT | 30.19 bp | 100.0% | 0.1 | 1.9M | 329.2% |
| 13 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 99.1% | 0.070 | AKEUSDT | 26.24 bp | 100.0% | 0 | 5.3M | 264.9% |
| 14 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 98.7% | 0.068 | AKEUSDT | 25.51 bp | 100.0% | 0 | 12.8M | 255.8% |
| 15 | SHAZ | SHAZUSDT (usdm-futures) | usdm-futures | 98.6% | 0.083 | ONEUSDT | 25.24 bp | 100.0% | 0.1 | 6.1M | 377.1% |
| 16 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 97.9% | 0.100 | BTCUSDT | 24.34 bp | 100.0% | 0 | 9.6M | 242.7% |
| 17 | DODO | DODOUSDT (spot) | spot | 99.0% | 0.074 | BLESSUSDT | 22.48 bp | 100.0% | 0 | 1.1M | 232.8% |
| 18 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 98.6% | 0.111 | BTTCUSDT | 21.52 bp | 100.0% | 0.2 | 1.4M | 228.5% |
| 19 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 97.5% | 0.156 | BTCUSDT | 19.89 bp | 100.0% | 0.1 | 1.4M | 231.1% |
| 20 | DGB | DGBUSDT (spot) | spot | 96.9% | 0.123 | RIFUSDT | 19.57 bp | 100.0% | 0.2 | 105.8K | 333.8% |
| 21 | SNXX | SNXXBUSDT (spot) | spot, usdm-futures | 96.6% | 0.197 | BTCUSDT | 18.93 bp | 100.0% | 0 | 390.1K | 273.4% |
| 22 | LAB | LABUSDT (usdm-futures) | usdm-futures | 96.5% | 0.121 | B2USDT | 23.42 bp | 100.0% | 0 | 19.9M | 263.0% |
| 23 | QNTB | QNTBUSDT (spot) | spot | 98.2% | 0.079 | ONEUSDT | 18.61 bp | 100.0% | 0.8 | 2.2K | 647.1% |
| 24 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 98.1% | 0.096 | ERAUSDT | 18.21 bp | 100.0% | 0 | 4.8M | 249.9% |
| 25 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 95.9% | 0.120 | LABUSDT | 17.64 bp | 100.0% | 0 | 1.5M | 203.8% |
| 26 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 96.5% | 0.111 | BTCUSDT | 17.53 bp | 100.0% | 0 | 982.2K | 176.0% |
| 27 | BAS | BASUSDT (usdm-futures) | usdm-futures | 96.9% | 0.121 | BTTCUSDT | 17.47 bp | 100.0% | 0 | 1.8M | 164.4% |
| 28 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 96.5% | 0.118 | BTCUSDT | 17.12 bp | 100.0% | 0.1 | 3.8M | 179.4% |
| 29 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 96.2% | 0.164 | ESPORTSUSDT | 16.30 bp | 100.0% | 0.2 | 985.9K | 227.3% |
| 30 | STAR | STARUSDT (usdm-futures) | usdm-futures | 97.2% | 0.092 | SAPIENUSDT | 16.08 bp | 100.0% | 0 | 414.6K | 176.9% |
| 31 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 97.5% | 0.078 | SNXXBUSDT | 15.77 bp | 100.0% | 0.1 | 695.3K | 182.0% |
| 32 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 95.8% | 0.128 | BTCUSDT | 15.59 bp | 100.0% | 0 | 4.4M | 168.6% |
| 33 | CAT | 1000CATUSDT (spot) | spot, usdm-futures | 95.9% | 0.107 | LABUSDT | 14.67 bp | 100.0% | 0.1 | 4K | 241.5% |
| 34 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 97.1% | 0.095 | 1000CATUSDT | 14.53 bp | 100.0% | 0 | 962.6K | 140.2% |
| 35 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 95.3% | 0.156 | BSBUSDT | 13.40 bp | 100.0% | 0 | 3.3M | 129.0% |
| 36 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 95.4% | 0.122 | EVAAUSDT | 13.28 bp | 100.0% | 0.1 | 4M | 129.5% |
| 37 | RAD | RADUSDT (spot) | spot | 95.9% | 0.116 | LABUSDT | 11.86 bp | 100.0% | 0.3 | 21.5K | 170.3% |
| 38 | ZHIPU | ZHIPUUSDT (usdm-futures) | usdm-futures | 95.4% | 0.103 | NIGHTUSDT | 11.67 bp | 100.0% | 0 | 4.4M | 128.0% |
| 39 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 95.3% | 0.099 | SAPIENUSDT | 11.45 bp | 100.0% | 0 | 240.2K | 119.8% |
| 40 | OPN | OPNUSDT (spot) | spot, usdm-futures | 94.6% | 0.171 | XNOUSDT | 14.52 bp | 100.0% | 0.1 | 2.4M | 161.0% |
| 41 | MBL | MBLUSDT (spot) | spot | 94.6% | 0.122 | MIRAUSDT | 12.91 bp | 100.0% | 0.1 | 73.2K | 132.3% |
| 42 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 94.7% | 0.148 | SNXXBUSDT | 11.22 bp | 100.0% | 0.1 | 639K | 210.8% |
| 43 | B | BUSDT (usdm-futures) | usdm-futures | 95.1% | 0.122 | MBLUSDT | 10.92 bp | 100.0% | 0.1 | 1.8M | 127.3% |
| 44 | WIN | WINUSDT (spot) | spot | 95.0% | 0.105 | PYRUSDT | 10.90 bp | 100.0% | 0.1 | 24.7K | 107.9% |
| 45 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 95.2% | 0.120 | BLESSUSDT | 10.38 bp | 100.0% | 0 | 246.9K | 101.3% |
| 46 | TKO | TKOUSDT (spot) | spot | 94.7% | 0.109 | BROCCOLIF3BUSDT | 9.30 bp | 100.0% | 0.3 | 16K | 126.1% |
| 47 | IBM | IBMBUSDT (spot) | spot, usdm-futures | 94.0% | 0.175 | ZHIPUUSDT | 11.56 bp | 100.0% | 0.1 | 666.6K | 189.8% |
| 48 | MINIMAX | MINIMAXUSDT (usdm-futures) | usdm-futures | 94.1% | 0.157 | IBMBUSDT | 9.03 bp | 100.0% | 0.1 | 845.7K | 129.0% |
| 49 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 94.0% | 0.146 | SNXXBUSDT | 10.62 bp | 100.0% | 0 | 9.1M | 185.1% |
| 50 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 94.0% | 0.153 | ERAUSDT | 8.94 bp | 100.0% | 0.1 | 1.2M | 126.1% |
| 51 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 94.9% | 0.145 | MINIMAXUSDT | 8.93 bp | 100.0% | 0 | 856.9K | 83.8% |
| 52 | NOK | NOKBUSDT (spot) | spot, usdm-futures | 94.4% | 0.106 | RADUSDT | 8.31 bp | 100.0% | 0.1 | 149.3K | 135.7% |
| 53 | TA | TAUSDT (usdm-futures) | usdm-futures | 93.9% | 0.149 | BTCUSDT | 7.84 bp | 100.0% | 0.1 | 455.1K | 81.6% |
| 54 | HEI | HEIUSDT (spot) | spot, usdm-futures | 93.8% | 0.105 | TAUSDT | 7.78 bp | 100.0% | 0.1 | 97.6K | 90.7% |
| 55 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 93.8% | 0.107 | NOKBUSDT | 7.28 bp | 100.0% | 0.1 | 111.2K | 76.0% |
| 56 | BAN | BANUSDT (usdm-futures) | usdm-futures | 93.8% | 0.108 | XNOUSDT | 6.48 bp | 100.0% | 0.1 | 2M | 114.6% |
| 57 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 93.4% | 0.137 | BSBUSDT | 8.29 bp | 100.0% | 0.1 | 157.5K | 79.3% |
| 58 | THE | THEUSDT (spot) | spot, usdm-futures | 93.0% | 0.117 | DEXEUSDT | 6.26 bp | 100.0% | 0.3 | 172.7K | 87.0% |
| 59 | AI | AIUSDT (spot) | spot | 93.7% | 0.154 | ZHIPUUSDT | 5.81 bp | 100.0% | 0.9 | 28.3K | 126.4% |
| 60 | GUN | GUNUSDT (spot) | spot, usdm-futures | 92.3% | 0.141 | BEATUSDT | 12.72 bp | 100.0% | 0.2 | 537.5K | 175.7% |
| 61 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 92.6% | 0.114 | BTCUSDT | 6.25 bp | 100.0% | 0.1 | 233.5K | 61.6% |
| 62 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 92.4% | 0.146 | BTCUSDT | 6.07 bp | 100.0% | 0.1 | 172.6K | 67.1% |
| 63 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 92.5% | 0.150 | BSBUSDT | 6.06 bp | 100.0% | 0.3 | 156.1K | 78.1% |
| 64 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 92.3% | 0.132 | BTCUSDT | 5.26 bp | 100.0% | 0 | 167.4K | 60.9% |
| 65 | REQ | REQUSDT (spot) | spot | 92.6% | 0.115 | NOKBUSDT | 4.64 bp | 100.0% | 0.5 | 8.3K | 85.4% |
| 66 | SOMI | SOMIUSDT (spot) | spot, usdm-futures | 92.5% | 0.106 | SPELLUSDT | 3.51 bp | 100.0% | 0.4 | 38.2K | 58.6% |
| 67 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 92.2% | 0.118 | ZAMAUSDT | 2.38 bp | 100.0% | 0.1 | 69.3K | 42.0% |
| 68 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 91.5% | 0.163 | STARUSDT | 8.12 bp | 100.0% | 0.1 | 377.6K | 84.6% |
| 69 | IOTA | IOTAUSDT (spot) | spot, usdm-futures | 91.0% | 0.147 | HEIUSDT | 7.46 bp | 100.0% | 0.4 | 136.5K | 110.9% |
| 70 | CITY | CITYUSDT (spot) | spot | 91.0% | 0.155 | BEATUSDT | 9.14 bp | 100.0% | 0.3 | 100.3K | 111.3% |
| 71 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 90.8% | 0.117 | QNTBUSDT | 7.64 bp | 100.0% | 0.2 | 886.3K | 91.8% |
| 72 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 91.2% | 0.121 | REQUSDT | 4.52 bp | 100.0% | 0.3 | 64.2K | 75.4% |
| 73 | AVGO | AVGOBUSDT (spot) | spot, usdm-futures | 91.0% | 0.129 | RIFUSDT | 4.16 bp | 100.0% | 0 | 15.4K | 82.9% |
| 74 | FWDI | FWDIUSDT (usdm-futures) | usdm-futures | 90.3% | 0.143 | BTCUSDT | 6.84 bp | 100.0% | 0.1 | 141.3K | 87.5% |
| 75 | ACH | ACHUSDT (spot) | spot, usdm-futures | 90.4% | 0.172 | BTCUSDT | 5.50 bp | 100.0% | 0.4 | 66.8K | 85.0% |
| 76 | NXPC | NXPCUSDT (spot) | spot, usdm-futures | 90.1% | 0.133 | BTCUSDT | 8.82 bp | 100.0% | 0.1 | 228.8K | 104.0% |
| 77 | YB | YBUSDT (spot) | spot, usdm-futures | 89.9% | 0.182 | BTCUSDT | 3.89 bp | 100.0% | 0.5 | 27.7K | 68.7% |
| 78 | SKL | SKLUSDT (spot) | spot, usdm-futures | 90.4% | 0.132 | HANAUSDT | 3.66 bp | 100.0% | 0.6 | 76.4K | 80.8% |
| 79 | SNOW | SNOWUSDT (usdm-futures) | usdm-futures | 89.9% | 0.135 | SHAZUSDT | 3.59 bp | 100.0% | 0 | 42.3K | 58.7% |
| 80 | ILV | ILVUSDT (spot) | spot, usdm-futures | 89.9% | 0.156 | BANUSDT | 3.33 bp | 100.0% | 0.5 | 13.4K | 78.6% |
| 81 | XAN | XANUSDT (usdm-futures) | usdm-futures | 89.2% | 0.133 | ZHIPUUSDT | 9.09 bp | 100.0% | 0 | 303.2K | 112.4% |
| 82 | SNX | SNXUSDT (spot) | spot, usdm-futures | 89.3% | 0.132 | TAKEUSDT | 8.73 bp | 100.0% | 0.6 | 520.8K | 152.3% |
| 83 | OPENAI | OPENAIUSDT (usdm-futures) | usdm-futures | 89.0% | 0.206 | ANTHROPICUSDT | 8.25 bp | 100.0% | 0 | 1.2M | 110.9% |
| 84 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 89.2% | 0.147 | AKEUSDT | 7.83 bp | 100.0% | 0.1 | 795.5K | 89.4% |
| 85 | GRAM | GRAMUSDT (spot) | spot, usdm-futures | 88.7% | 0.177 | BTCUSDT | 6.19 bp | 100.0% | 0.1 | 2.1M | 73.1% |
| 86 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 88.6% | 0.121 | DODOUSDT | 10.69 bp | 100.0% | 0 | 294.3K | 112.4% |
| 87 | PYTH | PYTHUSDT (spot) | spot, usdm-futures | 89.0% | 0.249 | BTCUSDT | 5.65 bp | 100.0% | 0.1 | 184.4K | 61.3% |
| 88 | ASML | ASMLUSDT (usdm-futures) | usdm-futures | 88.5% | 0.192 | SNXXBUSDT | 4.03 bp | 100.0% | 0 | 603.4K | 46.2% |
| 89 | SOFI | SOFIUSDT (usdm-futures) | usdm-futures | 89.1% | 0.134 | BTCUSDT | 3.33 bp | 100.0% | 0.1 | 89.3K | 46.6% |
| 90 | ID | IDUSDT (spot) | spot, usdm-futures | 88.8% | 0.132 | NAORISUSDT | 2.73 bp | 100.0% | 0.7 | 42.1K | 69.8% |
| 91 | VTHO | VTHOUSDT (spot) | spot, usdm-futures | 88.5% | 0.154 | BTCUSDT | 2.49 bp | 100.0% | 0.7 | 40.1K | 63.2% |
| 92 | HOT | HOTUSDT (spot) | spot, usdm-futures | 89.0% | 0.142 | GUAUSDT | 2.35 bp | 100.0% | 0.6 | 9.6K | 64.1% |
| 93 | WAXP | WAXPUSDT (spot) | spot, usdm-futures | 89.2% | 0.167 | SOFIUSDT | 2.02 bp | 100.0% | 0.6 | 6.4K | 55.2% |
| 94 | CATI | CATIUSDT (spot) | spot, usdm-futures | 88.6% | 0.154 | SOMIUSDT | 1.99 bp | 100.0% | 0.2 | 12.5K | 32.2% |
| 95 | AWE | AWEUSDT (spot) | spot, usdm-futures | 88.9% | 0.175 | TKOUSDT | 1.38 bp | 100.0% | 0.9 | 19.1K | 43.7% |
| 96 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 90.3% | 0.138 | WINUSDT | 1.28 bp | 100.0% | 0.1 | 44.4K | 25.9% |
| 97 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 86.9% | 0.197 | BTCUSDT | 6.35 bp | 100.0% | 0.1 | 128.6K | 73.7% |
| 98 | PROM | PROMUSDT (spot) | spot, usdm-futures | 85.8% | 0.192 | LABUSDT | 14.97 bp | 100.0% | 0.1 | 153K | 155.2% |
| 99 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 85.1% | 0.124 | FWDIUSDT | 14.52 bp | 100.0% | 0 | 2.1M | 165.7% |
| 100 | PIVX | PIVXUSDT (spot) | spot | 85.3% | 0.141 | SHAZUSDT | 11.93 bp | 100.0% | 0.3 | 67.6K | 187.4% |
| 101 | ONDS | ONDSUSDT (usdm-futures) | usdm-futures | 85.7% | 0.246 | SNXXBUSDT | 10.82 bp | 100.0% | 0.2 | 459.3K | 130.6% |
| 102 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 84.7% | 0.139 | EVAAUSDT | 13.52 bp | 100.0% | 0 | 499.2K | 153.5% |
| 103 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 85.4% | 0.202 | BTCUSDT | 10.74 bp | 100.0% | 0 | 1.3M | 105.6% |
| 104 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 84.6% | 0.147 | IDUSDT | 9.94 bp | 100.0% | 0 | 897.7K | 97.0% |
| 105 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 85.6% | 0.166 | BTCUSDT | 9.93 bp | 100.0% | 0 | 666.5K | 112.6% |
| 106 | ACM | ACMUSDT (spot) | spot | 85.0% | 0.160 | XNYUSDT | 9.04 bp | 100.0% | 0.3 | 24.1K | 133.9% |
| 107 | T | TUSDT (spot) | spot, usdm-futures | 84.5% | 0.187 | SPELLUSDT | 6.47 bp | 100.0% | 0.4 | 144.6K | 99.4% |
| 108 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 84.5% | 0.133 | RIFUSDT | 6.11 bp | 100.0% | 0.1 | 63.6K | 96.9% |
| 109 | MET | METUSDT (spot) | spot, usdm-futures | 84.2% | 0.230 | XPINUSDT | 6.32 bp | 100.0% | 0.2 | 146.2K | 95.0% |
| 110 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 84.1% | 0.150 | RADUSDT | 8.77 bp | 100.0% | 0.1 | 156.1K | 87.9% |
| 111 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 85.9% | 0.150 | IBMBUSDT | 5.83 bp | 100.0% | 0.1 | 109.1K | 75.4% |
| 112 | BAR | BARUSDT (spot) | spot | 83.8% | 0.139 | TRUTHUSDT | 7.22 bp | 100.0% | 0.4 | 69.7K | 118.0% |
| 113 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 83.6% | 0.167 | LABUSDT | 6.14 bp | 100.0% | 0.1 | 617.6K | 80.9% |
| 114 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 83.8% | 0.179 | BTCUSDT | 5.80 bp | 100.0% | 0.1 | 208.8K | 73.5% |
| 115 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 84.1% | 0.139 | 龙虾USDT | 5.73 bp | 100.0% | 0.1 | 42.2K | 58.4% |
| 116 | SYN | SYNUSDT (spot) | spot, usdm-futures | 83.0% | 0.128 | MIRAUSDT | 10.81 bp | 100.0% | 0 | 551.1K | 104.9% |
| 117 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 83.1% | 0.160 | DEXEUSDT | 8.84 bp | 100.0% | 0.1 | 81.9K | 89.5% |
| 118 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 82.9% | 0.134 | GRAMUSDT | 5.41 bp | 100.0% | 0.3 | 50.9K | 84.7% |
| 119 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 83.2% | 0.191 | BTCUSDT | 5.16 bp | 100.0% | 0.2 | 75.4K | 68.7% |
| 120 | GPS | GPSUSDT (spot) | spot, usdm-futures | 82.9% | 0.139 | TAKEUSDT | 5.14 bp | 100.0% | 0.2 | 49.9K | 70.9% |
| 121 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 82.3% | 0.137 | BILLUSDT | 6.11 bp | 100.0% | 0.1 | 63.1K | 62.1% |
| 122 | IRYS | IRYSUSDT (usdm-futures) | usdm-futures | 82.1% | 0.281 | BTCUSDT | 5.38 bp | 100.0% | 0.1 | 195.7K | 60.9% |
| 123 | TST | TSTUSDT (spot) | spot, usdm-futures | 84.1% | 0.134 | BEATUSDT | 5.14 bp | 100.0% | 0.3 | 76.7K | 98.4% |
| 124 | FLUID | FLUIDUSDT (usdm-futures) | usdm-futures | 81.6% | 0.144 | TAKEUSDT | 8.40 bp | 100.0% | 0.1 | 103.1K | 90.7% |
| 125 | OSMO | OSMOUSDT (spot) | spot | 81.4% | 0.131 | BROCCOLIF3BUSDT | 8.24 bp | 100.0% | 0.4 | 21.7K | 118.1% |
| 126 | Q | QUSDT (usdm-futures) | usdm-futures | 81.6% | 0.166 | BTCUSDT | 4.91 bp | 100.0% | 0.1 | 84.1K | 50.5% |
| 127 | PHA | PHAUSDT (spot) | spot, usdm-futures | 83.0% | 0.143 | ZHIPUUSDT | 4.84 bp | 100.0% | 0.6 | 157K | 107.9% |
| 128 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 80.5% | 0.215 | BTCUSDT | 8.02 bp | 100.0% | 0.1 | 97.6K | 90.4% |
| 129 | NOM | NOMUSDT (spot) | spot, usdm-futures | 80.4% | 0.158 | ANTHROPICUSDT | 8.06 bp | 100.0% | 0.8 | 139.6K | 176.2% |
| 130 | KSM | KSMUSDT (spot) | spot, usdm-futures | 80.3% | 0.185 | HANAUSDT | 4.42 bp | 100.0% | 0.4 | 23.6K | 86.9% |
| 131 | ZK | ZKUSDT (spot) | spot, usdm-futures | 80.2% | 0.140 | ILVUSDT | 5.14 bp | 100.0% | 0.1 | 149.3K | 68.0% |
| 132 | CIEN | CIENUSDT (usdm-futures) | usdm-futures | 82.1% | 0.143 | SHAZUSDT | 4.05 bp | 100.0% | 0 | 54.8K | 76.1% |
| 133 | TFUEL | TFUELUSDT (spot) | spot | 80.0% | 0.158 | THEUSDT | 6.81 bp | 100.0% | 0.2 | 55.4K | 104.6% |
| 134 | HK1810 | HK1810USDT (usdm-futures) | usdm-futures | 80.2% | 0.157 | ILVUSDT | 4.01 bp | 100.0% | 0.1 | 119.4K | 54.8% |
| 135 | WEN | WENUSDT (usdm-futures) | usdm-futures | 80.5% | 0.129 | XNOUSDT | 3.78 bp | 100.0% | 0.1 | 120K | 65.3% |
| 136 | WAL | WALUSDT (spot) | spot, usdm-futures | 80.1% | 0.213 | XANUSDT | 3.68 bp | 100.0% | 0.4 | 20.2K | 84.0% |
| 137 | TURTLE | TURTLEUSDT (spot) | spot, usdm-futures | 80.4% | 0.160 | BASUSDT | 3.52 bp | 100.0% | 0.7 | 41.3K | 77.5% |
| 138 | RIVN | RIVNUSDT (usdm-futures) | usdm-futures | 80.0% | 0.162 | QNTBUSDT | 3.48 bp | 100.0% | 0.1 | 164.6K | 59.1% |
| 139 | BZ | BZUSDT (usdm-futures) | usdm-futures | 79.8% | 0.151 | PYRUSDT | 3.44 bp | 100.0% | 0.1 | 82.8M | 36.9% |
| 140 | CC | CCUSDT (usdm-futures) | usdm-futures | 80.2% | 0.178 | BTCUSDT | 3.33 bp | 100.0% | 0 | 555.8K | 37.9% |
| 141 | O | OUSDT (usdm-futures) | usdm-futures | 79.3% | 0.179 | NOKBUSDT | 8.71 bp | 100.0% | 0 | 871.8K | 99.4% |
| 142 | MITO | MITOUSDT (spot) | spot, usdm-futures | 78.9% | 0.186 | SNXXBUSDT | 10.64 bp | 100.0% | 0.1 | 246.3K | 121.7% |
| 143 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 78.7% | 0.286 | BTCUSDT | 3.96 bp | 100.0% | 0 | 4.5M | 39.8% |
| 144 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 78.2% | 0.200 | BTCUSDT | 4.06 bp | 100.0% | 0.2 | 52.2K | 54.1% |
| 145 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 77.9% | 0.155 | BUSDT | 8.18 bp | 100.0% | 0.1 | 315.4K | 96.9% |
| 146 | ARM | ARMBUSDT (spot) | spot, usdm-futures | 78.3% | 0.142 | BULLAUSDT | 3.84 bp | 100.0% | 0 | 8.1K | 100.7% |
| 147 | BX | BXUSDT (usdm-futures) | usdm-futures | 78.4% | 0.151 | PIVXUSDT | 3.13 bp | 100.0% | 0 | 581K | 40.4% |
| 148 | YFI | YFIUSDT (spot) | spot, usdm-futures | 78.4% | 0.169 | BTCUSDT | 2.72 bp | 100.0% | 0.4 | 74.4K | 54.5% |
| 149 | AT | ATUSDT (spot) | spot, usdm-futures | 79.5% | 0.126 | ZBTUSDT | 2.54 bp | 100.0% | 0.3 | 20.6K | 48.6% |
| 150 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 79.6% | 0.141 | NVOUSDT | 2.43 bp | 100.0% | 0.1 | 2.9M | 31.8% |
| 151 | FORM | FORMUSDT (spot) | spot, usdm-futures | 77.4% | 0.214 | SNXXBUSDT | 2.37 bp | 100.0% | 0.5 | 38.2K | 40.9% |
| 152 | QKC | QKCUSDT (spot) | spot | 76.8% | 0.200 | FWDIUSDT | 5.89 bp | 100.0% | 0.2 | 7.1K | 87.1% |
| 153 | USTC | USTCUSDT (spot) | spot, usdm-futures | 76.6% | 0.141 | BANANAS31USDT | 2.98 bp | 100.0% | 0.8 | 24.9K | 68.6% |
| 154 | ICX | ICXUSDT (spot) | spot, usdm-futures | 77.1% | 0.189 | BUSDT | 2.13 bp | 100.0% | 1.3 | 12.9K | 86.3% |
| 155 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 76.4% | 0.217 | QNTBUSDT | 2.31 bp | 100.0% | 0 | 61.3K | 41.4% |
| 156 | SIGN | SIGNUSDT (spot) | spot, usdm-futures | 77.4% | 0.136 | FORMUSDT | 1.59 bp | 100.0% | 0.5 | 13K | 34.3% |
| 157 | NOW | NOWUSDT (usdm-futures) | usdm-futures | 75.8% | 0.183 | MINIMAXUSDT | 16.28 bp | 100.0% | 0 | 5.1M | 235.8% |
| 158 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 75.8% | 0.146 | AVGOBUSDT | 11.22 bp | 100.0% | 0.1 | 369.5K | 120.3% |
| 159 | GLMR | GLMRUSDT (spot) | spot | 75.8% | 0.136 | WAXPUSDT | 8.10 bp | 100.0% | 0.1 | 139.2K | 103.6% |
| 160 | HEMI | HEMIUSDT (spot) | spot, usdm-futures | 75.2% | 0.192 | BTCUSDT | 11.28 bp | 100.0% | 0.2 | 359.6K | 133.1% |
| 161 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 74.8% | 0.133 | WALUSDT | 11.13 bp | 100.0% | 0 | 289.3K | 106.6% |
| 162 | INX | INXUSDT (usdm-futures) | usdm-futures | 74.3% | 0.169 | ERAUSDT | 7.87 bp | 100.0% | 0.1 | 261K | 83.5% |
| 163 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 74.3% | 0.173 | 1000CATUSDT | 7.55 bp | 100.0% | 0.1 | 172.5K | 77.6% |
| 164 | JTO | JTOUSDT (spot) | spot, usdm-futures | 74.6% | 0.224 | BTCUSDT | 7.24 bp | 100.0% | 0 | 396.9K | 73.8% |
| 165 | SOPH | SOPHUSDT (spot) | spot, usdm-futures | 75.2% | 0.181 | SHAZUSDT | 3.69 bp | 100.0% | 0.4 | 14.3K | 69.6% |
| 166 | AXL | AXLUSDT (spot) | spot, usdm-futures | 74.8% | 0.171 | VTHOUSDT | 3.59 bp | 100.0% | 0.6 | 51.9K | 71.0% |
| 167 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 74.7% | 0.175 | SPACEUSDT | 3.16 bp | 100.0% | 0.1 | 416.7K | 42.6% |
| 168 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 74.0% | 0.226 | BTCUSDT | 3.24 bp | 100.0% | 0.2 | 49.2K | 42.6% |
| 169 | SATS | 1000SATSUSDT (spot) | spot, usdm-futures | 75.1% | 0.188 | MYXUSDT | 2.81 bp | 100.0% | 0.5 | 44.9K | 53.2% |
| 170 | LLY | LLYUSDT (usdm-futures) | usdm-futures | 75.8% | 0.119 | WAXPUSDT | 1.58 bp | 100.0% | 0.1 | 118.3K | 29.4% |
| 171 | PORTO | PORTOUSDT (spot) | spot | 72.2% | 0.128 | PYRUSDT | 15.46 bp | 100.0% | 0.1 | 44.2K | 177.3% |
| 172 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 71.7% | 0.157 | JTOUSDT | 12.72 bp | 100.0% | 0 | 483.5K | 147.7% |
| 173 | DIA | DIAUSDT (spot) | spot, usdm-futures | 71.8% | 0.209 | BTCUSDT | 10.80 bp | 100.0% | 0.3 | 63K | 112.1% |
| 174 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 71.6% | 0.272 | BTCUSDT | 9.85 bp | 100.0% | 0 | 708.2K | 95.9% |
| 175 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 72.0% | 0.118 | WENUSDT | 8.89 bp | 100.0% | 0.1 | 586.3K | 101.5% |
| 176 | BNC | BNCUSDT (usdm-futures) | usdm-futures | 71.9% | 0.144 | ERAUSDT | 5.50 bp | 100.0% | 0.2 | 80.9K | 73.6% |
| 177 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 71.6% | 0.201 | BTCUSDT | 5.26 bp | 100.0% | 0.3 | 53K | 75.1% |
| 178 | RAY | RAYUSDT (spot) | spot | 71.1% | 0.256 | BTCUSDT | 5.42 bp | 100.0% | 0.1 | 228K | 71.7% |
| 179 | REZ | REZUSDT (spot) | spot, usdm-futures | 72.2% | 0.192 | HEMIUSDT | 5.26 bp | 100.0% | 0.2 | 83.5K | 71.4% |
| 180 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 72.7% | 0.162 | SOPHUSDT | 4.91 bp | 100.0% | 0.2 | 71.6K | 72.8% |
| 181 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 70.5% | 0.235 | BTCUSDT | 7.91 bp | 100.0% | 0 | 586.2K | 78.3% |
| 182 | DCR | DCRUSDT (spot) | spot | 71.2% | 0.142 | ARMBUSDT | 4.71 bp | 100.0% | 0.4 | 68K | 70.2% |
| 183 | ORCL | ORCLBUSDT (spot) | spot, usdm-futures | 70.4% | 0.156 | BTTCUSDT | 3.31 bp | 100.0% | 0.1 | 32.5K | 78.1% |
| 184 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 71.5% | 0.175 | SNOWUSDT | 3.09 bp | 100.0% | 0 | 88.5K | 44.4% |
| 185 | MOCA | MOCAUSDT (usdm-futures) | usdm-futures | 70.1% | 0.158 | XMRUSDT | 4.93 bp | 100.0% | 0.3 | 35.6K | 59.1% |
| 186 | XAG | XAGUSDT (usdm-futures) | usdm-futures | 69.8% | 0.196 | BTCUSDT | 3.12 bp | 100.0% | 0.1 | 143.3M | 31.9% |
| 187 | ZKP | ZKPUSDT (spot) | spot, usdm-futures | 70.8% | 0.235 | LABUSDT | 2.85 bp | 100.0% | 0.8 | 39.3K | 58.3% |
| 188 | GAS | GASUSDT (spot) | spot, usdm-futures | 69.5% | 0.195 | IOTAUSDT | 2.21 bp | 100.0% | 0.4 | 16K | 41.0% |
| 189 | G | GUSDT (spot) | spot, usdm-futures | 69.8% | 0.179 | BUSDT | 1.74 bp | 100.0% | 1.1 | 40.7K | 53.4% |
| 190 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 69.0% | 0.174 | BTCUSDT | 12.29 bp | 100.0% | 0.1 | 468.2K | 131.6% |
| 191 | HD | HDUSDT (usdm-futures) | usdm-futures | 68.8% | 0.168 | ERAUSDT | 1.89 bp | 100.0% | 0 | 69.8K | 31.6% |
| 192 | MSFT | MSFTBUSDT (spot) | spot, usdm-futures | 68.6% | 0.141 | 币安人生USDT | 2.74 bp | 100.0% | 0.1 | 30.1K | 47.2% |
| 193 | BABA | BABABUSDT (spot) | spot, usdm-futures | 68.5% | 0.159 | QNTBUSDT | 2.07 bp | 100.0% | 0 | 8.6K | 42.1% |
| 194 | ARK | ARKUSDT (spot) | spot, usdm-futures | 68.7% | 0.179 | CCUSDT | 1.71 bp | 100.0% | 0.6 | 10.3K | 41.8% |
| 195 | CYBER | CYBERUSDT (spot) | spot, usdm-futures | 70.0% | 0.123 | WALUSDT | 1.65 bp | 100.0% | 1.7 | 34.1K | 55.5% |
| 196 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 67.4% | 0.161 | SPORTFUNUSDT | 10.32 bp | 100.0% | 0.1 | 463.2K | 109.7% |
| 197 | XEC | XECUSDT (spot) | spot, usdm-futures | 68.1% | 0.183 | SPORTFUNUSDT | 9.55 bp | 100.0% | 0.1 | 278.8K | 113.3% |
| 198 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 67.3% | 0.145 | FIGHTUSDT | 5.85 bp | 100.0% | 0.2 | 47.9K | 67.5% |
| 199 | MMT | MMTUSDT (spot) | spot, usdm-futures | 67.6% | 0.206 | STARUSDT | 5.75 bp | 100.0% | 0.1 | 136K | 77.1% |
| 200 | DOLO | DOLOUSDT (spot) | spot, usdm-futures | 67.4% | 0.162 | PORTALUSDT | 1.86 bp | 100.0% | 0.4 | 11.6K | 40.1% |
| 201 | RED | REDUSDT (spot) | spot, usdm-futures | 66.8% | 0.179 | HUMAUSDT | 3.75 bp | 100.0% | 0.4 | 62.2K | 64.0% |
| 202 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 66.5% | 0.203 | MYXUSDT | 5.14 bp | 100.0% | 0.1 | 179.6K | 55.3% |
| 203 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 66.6% | 0.153 | ARMBUSDT | 3.18 bp | 100.0% | 0.7 | 30.5K | 78.8% |
| 204 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 66.3% | 0.180 | PORTALUSDT | 13.33 bp | 100.0% | 0 | 1.9M | 131.7% |
| 205 | 我踏马来了 | 我踏马来了USDT (usdm-futures) | usdm-futures | 66.1% | 0.201 | GRAMUSDT | 5.42 bp | 100.0% | 0.1 | 218.4K | 60.0% |
| 206 | CTSI | CTSIUSDT (spot) | spot, usdm-futures | 67.3% | 0.184 | TSTUSDT | 1.20 bp | 100.0% | 1 | 4.3K | 34.7% |
| 207 | TRX | TRXUSDT (spot) | spot, usdm-futures, coinm-futures | 65.6% | 0.148 | GUSDT | 1.43 bp | 100.0% | 0.2 | 2.1M | 15.9% |
| 208 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 65.3% | 0.162 | AKEUSDT | 3.58 bp | 100.0% | 0.6 | 37K | 76.9% |
| 209 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 65.3% | 0.228 | SNXXBUSDT | 1.37 bp | 100.0% | 0 | 99.5K | 22.1% |
| 210 | UMA | UMAUSDT (spot) | spot, usdm-futures | 66.9% | 0.234 | AXLUSDT | 1.13 bp | 100.0% | 0.8 | 6.4K | 40.1% |
| 211 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 64.8% | 0.182 | BTCUSDT | 11.76 bp | 100.0% | 0.1 | 1.7M | 119.0% |
| 212 | ZM | ZMUSDT (usdm-futures) | usdm-futures | 64.4% | 0.174 | BSBUSDT | 3.46 bp | 100.0% | 0 | 70.1K | 53.0% |
| 213 | RPL | RPLUSDT (spot) | spot, usdm-futures | 64.2% | 0.174 | NIGHTUSDT | 7.50 bp | 100.0% | 1.2 | 27.1K | 152.1% |
| 214 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 63.8% | 0.189 | ATUSDT | 7.37 bp | 100.0% | 0.2 | 319.1K | 93.4% |
| 215 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 63.9% | 0.193 | GENIUSUSDT | 6.31 bp | 100.0% | 0.1 | 1.1M | 95.2% |
| 216 | HPE | HPEUSDT (usdm-futures) | usdm-futures | 63.3% | 0.205 | ONDSUSDT | 6.02 bp | 100.0% | 0 | 225K | 80.7% |
| 217 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 63.0% | 0.163 | BARUSDT | 7.62 bp | 100.0% | 0.6 | 253.8K | 129.1% |
| 218 | GLW | GLWBUSDT (spot) | spot, usdm-futures | 63.3% | 0.358 | SNXXBUSDT | 4.32 bp | 100.0% | 0 | 126.6K | 84.6% |
| 219 | IOTX | IOTXUSDT (spot) | spot, usdm-futures | 63.1% | 0.146 | PYTHUSDT | 1.95 bp | 100.0% | 0.9 | 8.9K | 65.1% |
| 220 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 62.6% | 0.182 | IRYSUSDT | 7.41 bp | 100.0% | 0.1 | 617.1K | 78.3% |
| 221 | PIXEL | PIXELUSDT (spot) | spot, usdm-futures | 62.5% | 0.245 | BTCUSDT | 3.00 bp | 100.0% | 0.6 | 20.1K | 67.0% |
| 222 | V | VUSDT (usdm-futures) | usdm-futures | 64.0% | 0.150 | BZUSDT | 1.23 bp | 100.0% | 0 | 54.6K | 21.8% |
| 223 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 60.8% | 0.152 | HEIUSDT | 15.70 bp | 100.0% | 0 | 3.3M | 157.8% |
| 224 | UB | UBUSDT (usdm-futures) | usdm-futures | 61.4% | 0.149 | LABUSDT | 15.19 bp | 100.0% | 0 | 4.6M | 147.8% |
| 225 | ACE | ACEUSDT (spot) | spot, usdm-futures | 61.0% | 0.159 | ACHUSDT | 14.90 bp | 100.0% | 0.1 | 386.8K | 155.8% |
| 226 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 60.1% | 0.202 | CITYUSDT | 10.31 bp | 100.0% | 0.1 | 2.2M | 104.6% |
| 227 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 61.2% | 0.157 | TSTUSDT | 9.42 bp | 100.0% | 0.1 | 278.3K | 114.3% |
| 228 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 60.1% | 0.146 | FORMUSDT | 8.30 bp | 100.0% | 0.1 | 256.6K | 93.2% |
| 229 | WET | WETUSDT (usdm-futures) | usdm-futures | 60.2% | 0.158 | AXLUSDT | 5.18 bp | 100.0% | 0.1 | 199K | 62.6% |
| 230 | RIVER | RIVERUSDT (usdm-futures) | usdm-futures | 59.9% | 0.317 | BTCUSDT | 6.88 bp | 100.0% | 0.1 | 1.8M | 70.9% |
| 231 | JOE | JOEUSDT (spot) | spot, usdm-futures | 60.4% | 0.163 | PYRUSDT | 4.99 bp | 100.0% | 0.6 | 30.4K | 108.4% |
| 232 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 60.6% | 0.184 | 1000SATSUSDT | 4.36 bp | 100.0% | 0.2 | 96.5K | 60.9% |
| 233 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 60.0% | 0.219 | HANAUSDT | 2.49 bp | 100.0% | 0 | 186K | 42.4% |
| 234 | STG | STGUSDT (spot) | spot, usdm-futures | 59.6% | 0.157 | CYBERUSDT | 2.31 bp | 100.0% | 0.4 | 34.7K | 45.8% |
| 235 | COMP | COMPUSDT (spot) | spot, usdm-futures | 59.8% | 0.333 | MYXUSDT | 2.07 bp | 100.0% | 0.7 | 28.5K | 43.2% |
| 236 | MTL | MTLUSDT (spot) | spot, usdm-futures | 60.1% | 0.262 | OUSDT | 1.11 bp | 100.0% | 1.6 | 2.7K | 56.3% |
| 237 | METIS | METISUSDT (spot) | spot, usdm-futures | 59.0% | 0.157 | WAXPUSDT | 1.10 bp | 100.0% | 1.8 | 9.2K | 45.5% |
| 238 | STRC | STRCUSDT (usdm-futures) | usdm-futures | 60.2% | 0.202 | HPEUSDT | 1.00 bp | 100.0% | 0.1 | 274.5K | 18.9% |
| 239 | MVLL | MVLLBUSDT (spot) | spot, usdm-futures | 58.1% | 0.253 | REZUSDT | 13.00 bp | 100.0% | 0 | 14K | 280.8% |
| 240 | MON | MONUSDT (usdm-futures) | usdm-futures | 57.7% | 0.346 | BTCUSDT | 7.54 bp | 100.0% | 0.1 | 2.3M | 78.6% |
| 241 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 57.7% | 0.156 | CHIPUSDT | 6.74 bp | 100.0% | 0 | 448.9K | 70.7% |
| 242 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 56.8% | 0.190 | ANTHROPICUSDT | 6.71 bp | 100.0% | 0.1 | 279K | 68.9% |
| 243 | M | MUSDT (usdm-futures) | usdm-futures | 57.4% | 0.195 | TAKEUSDT | 6.22 bp | 100.0% | 0.1 | 517K | 64.1% |
| 244 | ORDER | ORDERUSDT (usdm-futures) | usdm-futures | 56.5% | 0.259 | BTCUSDT | 6.01 bp | 100.0% | 0.1 | 281.3K | 83.9% |
| 245 | TUT | TUTUSDT (spot) | spot, usdm-futures | 56.4% | 0.139 | BRKBUSDT | 6.19 bp | 100.0% | 0.2 | 93.1K | 85.0% |
| 246 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 57.4% | 0.181 | JELLYJELLYUSDT | 5.74 bp | 100.0% | 0.1 | 503K | 86.5% |
| 247 | KERNEL | KERNELUSDT (spot) | spot, usdm-futures | 55.9% | 0.153 | MYXUSDT | 5.39 bp | 100.0% | 0.7 | 32.9K | 99.8% |
| 248 | MELANIA | MELANIAUSDT (usdm-futures) | usdm-futures | 56.9% | 0.174 | BANANAS31USDT | 4.67 bp | 100.0% | 0.1 | 109.2K | 54.1% |
| 249 | NEXO | NEXOUSDT (spot) | spot | 55.5% | 0.168 | MANTAUSDT | 7.02 bp | 100.0% | 0.2 | 184.8K | 87.8% |
| 250 | OPEN | OPENUSDT (spot) | spot, usdm-futures | 55.4% | 0.197 | BTCUSDT | 5.07 bp | 100.0% | 0.3 | 80.7K | 67.8% |
| 251 | TAC | TACUSDT (usdm-futures) | usdm-futures | 55.1% | 0.176 | LABUSDT | 14.20 bp | 100.0% | 0 | 1.2M | 151.6% |
| 252 | ETHW | ETHWUSDT (usdm-futures) | usdm-futures | 55.8% | 0.206 | COMPUSDT | 4.65 bp | 100.0% | 0.1 | 44.6K | 53.7% |
| 253 | KAT | KATUSDT (spot) | spot, usdm-futures | 55.3% | 0.176 | DOLOUSDT | 3.25 bp | 100.0% | 0.4 | 40.7K | 63.9% |
| 254 | USELESS | USELESSUSDT (usdm-futures) | usdm-futures | 54.2% | 0.244 | ONDSUSDT | 10.00 bp | 100.0% | 0 | 1.3M | 144.3% |
| 255 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 53.9% | 0.180 | FWDIUSDT | 7.82 bp | 100.0% | 0 | 383.8K | 80.7% |
| 256 | ALICE | ALICEUSDT (spot) | spot, usdm-futures | 54.0% | 0.161 | HMSTRUSDT | 5.21 bp | 100.0% | 0.3 | 81.6K | 65.0% |
| 257 | JUV | JUVUSDT (spot) | spot | 53.3% | 0.175 | ACMUSDT | 7.92 bp | 100.0% | 0.3 | 45.2K | 113.7% |

## Diagnostics

- Basis size selected: 257
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.044
- Maximum pairwise absolute correlation: 0.358
- Mean whole-market projection R²: 85.9%
- Median whole-market projection R²: 80.3%
- 10th-percentile whole-market projection R²: 74.3%
- Minimum whole-market projection R²: 70.6%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.7% | 1.4% | 0.0% | 0.0% |
| 5 | 6.4% | 2.9% | 0.7% | 0.1% |
| 10 | 8.5% | 4.7% | 1.9% | 0.5% |
| 15 | 10.6% | 6.3% | 3.2% | 1.0% |
| 20 | 12.7% | 7.9% | 4.5% | 2.0% |
| 25 | 15.6% | 10.0% | 5.8% | 3.4% |
| 30 | 17.6% | 11.4% | 7.2% | 4.5% |
| 35 | 19.6% | 13.2% | 8.7% | 5.0% |
| 40 | 21.5% | 14.6% | 9.9% | 6.0% |
| 45 | 23.5% | 16.2% | 11.4% | 6.8% |
| 50 | 25.3% | 17.6% | 12.7% | 7.3% |
| 55 | 27.3% | 19.5% | 14.2% | 8.0% |
| 60 | 29.3% | 21.3% | 15.7% | 10.4% |
| 65 | 31.2% | 22.9% | 17.1% | 10.7% |
| 70 | 33.2% | 24.6% | 19.0% | 13.3% |
| 75 | 35.1% | 26.2% | 20.6% | 14.8% |
| 80 | 37.0% | 27.9% | 22.0% | 16.3% |
| 85 | 38.7% | 29.3% | 23.5% | 17.5% |
| 90 | 40.7% | 31.3% | 25.2% | 17.9% |
| 95 | 42.5% | 32.8% | 26.8% | 18.5% |
| 100 | 44.1% | 34.4% | 28.2% | 23.9% |
| 105 | 45.9% | 36.2% | 29.7% | 25.0% |
| 110 | 47.5% | 37.6% | 31.0% | 26.1% |
| 115 | 49.1% | 38.9% | 32.5% | 27.7% |
| 120 | 50.8% | 40.7% | 34.2% | 28.9% |
| 125 | 52.5% | 42.5% | 36.0% | 30.6% |
| 130 | 54.1% | 44.0% | 37.5% | 32.3% |
| 135 | 55.7% | 45.7% | 38.8% | 32.9% |
| 140 | 57.2% | 47.0% | 40.3% | 34.3% |
| 145 | 58.6% | 48.5% | 41.8% | 36.4% |
| 150 | 60.2% | 50.0% | 43.3% | 37.5% |
| 155 | 61.7% | 51.5% | 44.7% | 38.6% |
| 160 | 63.1% | 53.1% | 46.2% | 41.6% |
| 165 | 64.6% | 54.8% | 47.8% | 42.3% |
| 170 | 66.0% | 56.3% | 49.3% | 45.7% |
| 175 | 67.3% | 57.9% | 50.9% | 46.5% |
| 180 | 68.7% | 59.4% | 52.5% | 47.7% |
| 185 | 70.0% | 60.8% | 54.1% | 49.0% |
| 190 | 71.4% | 62.5% | 55.5% | 50.4% |
| 195 | 72.7% | 64.0% | 56.9% | 52.2% |
| 200 | 73.9% | 65.4% | 58.2% | 53.0% |
| 205 | 75.0% | 66.9% | 59.7% | 54.5% |
| 210 | 76.3% | 68.4% | 61.1% | 56.3% |
| 215 | 77.4% | 69.7% | 62.7% | 58.1% |
| 220 | 78.6% | 71.0% | 64.2% | 59.1% |
| 225 | 79.6% | 72.1% | 65.5% | 61.9% |
| 230 | 80.7% | 73.6% | 66.8% | 62.6% |
| 235 | 81.7% | 74.8% | 68.3% | 62.9% |
| 240 | 82.7% | 75.9% | 69.7% | 66.2% |
| 245 | 83.7% | 77.4% | 71.0% | 67.0% |
| 250 | 84.6% | 78.4% | 72.3% | 68.2% |
| 255 | 85.5% | 79.8% | 73.8% | 70.1% |
| 257 | 85.9% | 80.3% | 74.3% | 70.6% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | XNO | BTTC | RIF | B2 | BANK | PYR | ONE | DEXE | AKE | ON | ERA | BLESS | BROCCOLIF3B | SHAZ | ESPORTS | DODO | MIRA | SAPIEN | DGB | SNXX | LAB | QNTB | AIA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.058 | 0.059 | -0.034 | -0.051 | 0.138 | -0.052 | -0.061 | 0.010 | 0.032 | 0.044 | 0.091 | -0.024 | -0.035 | -0.019 | 0.100 | 0.009 | 0.039 | 0.156 | 0.120 | 0.197 | 0.028 | -0.025 | -0.002 |
| XNO | 0.058 | 1.000 | 0.050 | -0.075 | -0.005 | 0.053 | -0.043 | -0.053 | -0.040 | 0.011 | 0.004 | -0.036 | -0.059 | 0.009 | -0.009 | -0.035 | 0.010 | 0.003 | 0.059 | -0.000 | 0.019 | 0.056 | 0.002 | 0.043 |
| BTTC | 0.059 | 0.050 | 1.000 | -0.015 | 0.032 | -0.033 | -0.055 | 0.009 | 0.077 | -0.095 | -0.080 | 0.023 | -0.021 | 0.046 | -0.071 | 0.018 | -0.061 | 0.111 | 0.013 | -0.012 | 0.031 | -0.008 | 0.013 | 0.050 |
| RIF | -0.034 | -0.075 | -0.015 | 1.000 | 0.000 | -0.018 | 0.022 | 0.006 | -0.042 | 0.002 | -0.011 | 0.024 | 0.015 | 0.059 | -0.030 | -0.020 | -0.001 | -0.020 | -0.011 | 0.123 | -0.030 | 0.112 | -0.036 | 0.006 |
| B2 | -0.051 | -0.005 | 0.032 | 0.000 | 1.000 | 0.032 | -0.091 | -0.035 | -0.029 | -0.046 | -0.055 | 0.010 | -0.015 | 0.005 | -0.005 | 0.000 | 0.012 | -0.042 | 0.023 | -0.074 | 0.025 | 0.121 | 0.011 | 0.006 |
| BANK | 0.138 | 0.053 | -0.033 | -0.018 | 0.032 | 1.000 | -0.050 | -0.070 | -0.019 | 0.012 | 0.016 | 0.021 | -0.051 | 0.060 | -0.054 | 0.077 | -0.008 | -0.016 | 0.055 | -0.016 | 0.004 | 0.040 | 0.041 | 0.011 |
| PYR | -0.052 | -0.043 | -0.055 | 0.022 | -0.091 | -0.050 | 1.000 | -0.000 | -0.059 | -0.014 | -0.024 | -0.066 | -0.040 | -0.042 | 0.013 | 0.012 | 0.003 | -0.065 | 0.010 | -0.031 | -0.032 | 0.108 | 0.013 | 0.038 |
| ONE | -0.061 | -0.053 | 0.009 | 0.006 | -0.035 | -0.070 | -0.000 | 1.000 | 0.031 | -0.008 | -0.105 | 0.124 | -0.020 | 0.002 | -0.083 | -0.044 | 0.042 | -0.023 | -0.026 | -0.042 | 0.031 | -0.088 | -0.079 | -0.068 |
| DEXE | 0.010 | -0.040 | 0.077 | -0.042 | -0.029 | -0.019 | -0.059 | 0.031 | 1.000 | -0.012 | 0.064 | -0.028 | -0.019 | 0.046 | -0.020 | -0.065 | 0.021 | 0.046 | 0.023 | -0.063 | -0.011 | -0.059 | 0.060 | -0.025 |
| AKE | 0.032 | 0.011 | -0.095 | 0.002 | -0.046 | 0.012 | -0.014 | -0.008 | -0.012 | 1.000 | -0.042 | 0.035 | 0.070 | -0.068 | 0.016 | 0.000 | 0.051 | 0.015 | -0.053 | 0.074 | 0.029 | -0.029 | 0.071 | 0.003 |
| ON | 0.044 | 0.004 | -0.080 | -0.011 | -0.055 | 0.016 | -0.024 | -0.105 | 0.064 | -0.042 | 1.000 | 0.125 | -0.004 | -0.054 | 0.081 | 0.063 | 0.008 | -0.021 | 0.010 | 0.108 | -0.036 | -0.008 | -0.054 | -0.019 |
| ERA | 0.091 | -0.036 | 0.023 | 0.024 | 0.010 | 0.021 | -0.066 | 0.124 | -0.028 | 0.035 | 0.125 | 1.000 | -0.055 | -0.063 | 0.030 | -0.024 | 0.023 | 0.001 | 0.041 | 0.058 | 0.068 | -0.003 | 0.034 | -0.096 |
| BLESS | -0.024 | -0.059 | -0.021 | 0.015 | -0.015 | -0.051 | -0.040 | -0.020 | -0.019 | 0.070 | -0.004 | -0.055 | 1.000 | 0.027 | 0.054 | 0.013 | 0.074 | -0.050 | -0.047 | 0.018 | 0.010 | -0.001 | -0.034 | -0.049 |
| BROCCOLIF3B | -0.035 | 0.009 | 0.046 | 0.059 | 0.005 | 0.060 | -0.042 | 0.002 | 0.046 | -0.068 | -0.054 | -0.063 | 0.027 | 1.000 | 0.016 | 0.096 | 0.010 | 0.023 | -0.015 | -0.024 | -0.084 | -0.006 | 0.021 | 0.025 |
| SHAZ | -0.019 | -0.009 | -0.071 | -0.030 | -0.005 | -0.054 | 0.013 | -0.083 | -0.020 | 0.016 | 0.081 | 0.030 | 0.054 | 0.016 | 1.000 | 0.062 | 0.004 | -0.015 | -0.054 | -0.005 | -0.003 | -0.000 | -0.012 | 0.046 |
| ESPORTS | 0.100 | -0.035 | 0.018 | -0.020 | 0.000 | 0.077 | 0.012 | -0.044 | -0.065 | 0.000 | 0.063 | -0.024 | 0.013 | 0.096 | 0.062 | 1.000 | 0.063 | -0.037 | -0.024 | -0.015 | -0.048 | 0.081 | -0.017 | 0.092 |
| DODO | 0.009 | 0.010 | -0.061 | -0.001 | 0.012 | -0.008 | 0.003 | 0.042 | 0.021 | 0.051 | 0.008 | 0.023 | 0.074 | 0.010 | 0.004 | 0.063 | 1.000 | -0.035 | -0.026 | 0.020 | -0.040 | 0.067 | -0.014 | -0.013 |
| MIRA | 0.039 | 0.003 | 0.111 | -0.020 | -0.042 | -0.016 | -0.065 | -0.023 | 0.046 | 0.015 | -0.021 | 0.001 | -0.050 | 0.023 | -0.015 | -0.037 | -0.035 | 1.000 | 0.098 | 0.023 | -0.059 | 0.017 | 0.005 | -0.025 |
| SAPIEN | 0.156 | 0.059 | 0.013 | -0.011 | 0.023 | 0.055 | 0.010 | -0.026 | 0.023 | -0.053 | 0.010 | 0.041 | -0.047 | -0.015 | -0.054 | -0.024 | -0.026 | 0.098 | 1.000 | 0.017 | 0.082 | 0.040 | -0.032 | 0.006 |
| DGB | 0.120 | -0.000 | -0.012 | 0.123 | -0.074 | -0.016 | -0.031 | -0.042 | -0.063 | 0.074 | 0.108 | 0.058 | 0.018 | -0.024 | -0.005 | -0.015 | 0.020 | 0.023 | 0.017 | 1.000 | 0.031 | 0.001 | -0.039 | -0.003 |
| SNXX | 0.197 | 0.019 | 0.031 | -0.030 | 0.025 | 0.004 | -0.032 | 0.031 | -0.011 | 0.029 | -0.036 | 0.068 | 0.010 | -0.084 | -0.003 | -0.048 | -0.040 | -0.059 | 0.082 | 0.031 | 1.000 | -0.008 | -0.051 | -0.066 |
| LAB | 0.028 | 0.056 | -0.008 | 0.112 | 0.121 | 0.040 | 0.108 | -0.088 | -0.059 | -0.029 | -0.008 | -0.003 | -0.001 | -0.006 | -0.000 | 0.081 | 0.067 | 0.017 | 0.040 | 0.001 | -0.008 | 1.000 | 0.041 | 0.006 |
| QNTB | -0.025 | 0.002 | 0.013 | -0.036 | 0.011 | 0.041 | 0.013 | -0.079 | 0.060 | 0.071 | -0.054 | 0.034 | -0.034 | 0.021 | -0.012 | -0.017 | -0.014 | 0.005 | -0.032 | -0.039 | -0.051 | 0.041 | 1.000 | 0.039 |
| AIA | -0.002 | 0.043 | 0.050 | 0.006 | 0.006 | 0.011 | 0.038 | -0.068 | -0.025 | 0.003 | -0.019 | -0.096 | -0.049 | 0.025 | 0.046 | 0.092 | -0.013 | -0.025 | 0.006 | -0.003 | -0.066 | 0.006 | 0.039 | 1.000 |

The complete 257 × 257 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| C | CUSDT | 70.6% | 54.2% | JTOUSDT | 0.176 |
| POLYX | POLYXUSDT | 70.7% | 54.2% | ILVUSDT | 0.175 |
| SUN | SUNUSDT | 70.8% | 54.1% | AIOTUSDT | 0.173 |
| FLEX | FLEXUSDT | 71.0% | 53.9% | ICXUSDT | 0.241 |
| POWR | POWRUSDT | 71.1% | 53.8% | REZUSDT | 0.165 |
| FOGO | FOGOUSDT | 71.1% | 53.7% | ICNTUSDT | 0.177 |
| NMR | NMRUSDT | 71.3% | 53.6% | BANUSDT | -0.241 |
| EGLD | EGLDUSDT | 71.4% | 53.5% | NEXOUSDT | 0.168 |
| AUCTION | AUCTIONUSDT | 71.5% | 53.4% | SIGNUSDT | 0.196 |
| BAT | BATUSDT | 71.8% | 53.1% | DOLOUSDT | 0.186 |
| CKB | CKBUSDT | 71.8% | 53.1% | USTCUSDT | 0.222 |
| RSR | RSRUSDT | 71.8% | 53.1% | WAXPUSDT | 0.221 |
| RLC | RLCUSDT | 71.8% | 53.1% | SIGNUSDT | 0.254 |
| XVG | XVGUSDT | 71.9% | 53.0% | ARKUSDT | 0.194 |
| MUBARAK | MUBARAKUSDT | 72.1% | 52.8% | BTCUSDT | 0.235 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

