# Binance portfolio basis

Generated 2026-07-23T19:17:13.097Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-08T00:00Z through 2026-07-22T23:00Z
- Sampling: exactly 360 1h log returns (15 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 5.0% of the maximum unexplained variance, select the largest mean absolute 1h return
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
| 2 | LAB | LABUSDT (usdm-futures) | usdm-futures | 98.7% | 0.160 | BTCUSDT | 382.94 bp | 100.0% | 0 | 270.1M | 583.2% |
| 3 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 99.5% | 0.073 | LABUSDT | 377.08 bp | 100.0% | 0 | 219.8M | 935.5% |
| 4 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 99.0% | 0.103 | BTCUSDT | 344.21 bp | 100.0% | 1 | 83.9M | 551.2% |
| 5 | BANK | BANKUSDT (spot) | spot, usdm-futures | 99.6% | 0.069 | ESPORTSUSDT | 285.54 bp | 100.0% | 3 | 2.5M | 589.7% |
| 6 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.7% | 0.048 | LABUSDT | 282.37 bp | 100.0% | 1 | 185.8M | 479.2% |
| 7 | B | BUSDT (usdm-futures) | usdm-futures | 98.9% | 0.113 | EVAAUSDT | 234.77 bp | 100.0% | 1 | 24.3M | 400.1% |
| 8 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 98.1% | 0.162 | LABUSDT | 222.99 bp | 100.0% | 1 | 9.7M | 445.2% |
| 9 | US | USUSDT (usdm-futures) | usdm-futures | 98.5% | 0.114 | ESPORTSUSDT | 209.81 bp | 100.0% | 0 | 29.4M | 306.6% |
| 10 | TLM | TLMUSDT (spot) | spot, usdm-futures | 99.1% | 0.080 | BANKUSDT | 202.82 bp | 100.0% | 1 | 4M | 315.8% |
| 11 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 98.2% | 0.116 | TAGUSDT | 196.90 bp | 100.0% | 1 | 28.2M | 310.1% |
| 12 | SYN | SYNUSDT (spot) | spot, usdm-futures | 98.5% | 0.107 | LABUSDT | 191.68 bp | 100.0% | 0 | 7.2M | 273.8% |
| 13 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 98.1% | 0.124 | TLMUSDT | 180.81 bp | 100.0% | 0 | 9.2M | 245.5% |
| 14 | DODO | DODOUSDT (spot) | spot | 99.7% | 0.053 | ESPORTSUSDT | 172.04 bp | 100.0% | 1 | 3.3M | 319.7% |
| 15 | ON | ONUSDT (usdm-futures) | usdm-futures | 98.8% | 0.090 | LABUSDT | 169.16 bp | 100.0% | 1 | 2.4M | 256.1% |
| 16 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 97.5% | 0.104 | LABUSDT | 168.34 bp | 100.0% | 1 | 7.4M | 298.4% |
| 17 | RIF | RIFUSDT (spot) | spot, usdm-futures | 98.1% | 0.108 | TLMUSDT | 162.95 bp | 100.0% | 2 | 1.8M | 246.3% |
| 18 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 97.8% | 0.088 | VELVETUSDT | 160.75 bp | 100.0% | 0 | 11.9M | 241.4% |
| 19 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 96.4% | 0.157 | ESPORTSUSDT | 155.83 bp | 100.0% | 0 | 3.5M | 254.1% |
| 20 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 97.7% | 0.122 | BTCUSDT | 154.47 bp | 100.0% | 2 | 12.7M | 195.4% |
| 21 | PYR | PYRUSDT (spot) | spot | 96.6% | 0.193 | BUSDT | 151.47 bp | 100.0% | 3 | 1.1M | 267.1% |
| 22 | BTTC | BTTCUSDT (spot) | spot | 98.1% | 0.101 | CLOUSDT | 145.78 bp | 100.0% | 17 | 139.3K | 220.6% |
| 23 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.8% | 0.114 | BTCUSDT | 143.30 bp | 100.0% | 1 | 4.4M | 214.2% |
| 24 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 98.4% | 0.080 | CLOUSDT | 138.80 bp | 100.0% | 1 | 6.8M | 199.7% |
| 25 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 95.9% | 0.204 | LABUSDT | 142.66 bp | 100.0% | 1 | 20.8M | 191.7% |
| 26 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 96.9% | 0.118 | EVAAUSDT | 136.58 bp | 100.0% | 1 | 2.6M | 225.5% |
| 27 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 96.7% | 0.168 | USUSDT | 133.78 bp | 100.0% | 1 | 7.1M | 236.9% |
| 28 | XEC | XECUSDT (spot) | spot, usdm-futures | 97.0% | 0.122 | AIOTUSDT | 132.06 bp | 100.0% | 2 | 2.4M | 218.4% |
| 29 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 95.3% | 0.130 | LABUSDT | 134.78 bp | 100.0% | 2 | 2.1M | 170.6% |
| 30 | DGB | DGBUSDT (spot) | spot | 95.3% | 0.169 | BTTCUSDT | 131.76 bp | 100.0% | 3 | 114.2K | 229.6% |
| 31 | SXT | SXTUSDT (spot) | spot, usdm-futures | 95.5% | 0.123 | DODOUSDT | 130.75 bp | 100.0% | 2 | 4.1M | 200.9% |
| 32 | HEI | HEIUSDT (spot) | spot, usdm-futures | 97.0% | 0.095 | TLMUSDT | 127.92 bp | 100.0% | 1 | 1.9M | 160.2% |
| 33 | XNO | XNOUSDT (spot) | spot | 97.0% | 0.103 | ONUSDT | 126.64 bp | 100.0% | 3 | 43.1K | 320.2% |
| 34 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 95.2% | 0.124 | TAGUSDT | 126.59 bp | 100.0% | 1 | 4.4M | 216.1% |
| 35 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 95.0% | 0.166 | PYRUSDT | 127.96 bp | 100.0% | 0 | 3.8M | 215.5% |
| 36 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 96.4% | 0.116 | BANKUSDT | 125.42 bp | 100.0% | 1 | 2.7M | 188.4% |
| 37 | SKL | SKLUSDT (spot) | spot, usdm-futures | 95.7% | 0.105 | BTTCUSDT | 117.72 bp | 100.0% | 4 | 1.1M | 198.4% |
| 38 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 95.3% | 0.115 | BTCUSDT | 111.94 bp | 100.0% | 1 | 21.5M | 147.9% |
| 39 | PORTO | PORTOUSDT (spot) | spot | 94.6% | 0.133 | TLMUSDT | 111.13 bp | 100.0% | 2 | 526.7K | 244.8% |
| 40 | ACE | ACEUSDT (spot) | spot, usdm-futures | 95.5% | 0.113 | TLMUSDT | 110.95 bp | 100.0% | 2 | 170.2K | 251.9% |
| 41 | ATM | ATMUSDT (spot) | spot | 96.3% | 0.120 | SKLUSDT | 108.54 bp | 100.0% | 1 | 1.2M | 252.5% |
| 42 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 94.5% | 0.138 | BTCUSDT | 105.87 bp | 100.0% | 2 | 2.2M | 158.8% |
| 43 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 94.8% | 0.095 | PARTIUSDT | 104.48 bp | 100.0% | 0 | 2.7M | 211.0% |
| 44 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 94.9% | 0.136 | LABUSDT | 97.00 bp | 100.0% | 1 | 2.4M | 182.0% |
| 45 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 94.6% | 0.146 | PARTIUSDT | 86.81 bp | 100.0% | 1 | 1.9M | 135.9% |
| 46 | PROM | PROMUSDT (spot) | spot, usdm-futures | 94.0% | 0.209 | BANKUSDT | 116.68 bp | 100.0% | 2 | 300.4K | 211.7% |
| 47 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 94.1% | 0.162 | CLOUSDT | 104.10 bp | 100.0% | 1 | 1.4M | 230.9% |
| 48 | TREE | TREEUSDT (spot) | spot, usdm-futures | 94.6% | 0.156 | BTCUSDT | 83.47 bp | 100.0% | 3 | 6.7M | 160.3% |
| 49 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 94.0% | 0.145 | BEATUSDT | 79.83 bp | 100.0% | 1 | 1.3M | 99.8% |
| 50 | B2 | B2USDT (usdm-futures) | usdm-futures | 94.1% | 0.113 | SLXUSDT | 70.67 bp | 100.0% | 1 | 916.5K | 123.3% |
| 51 | BAR | BARUSDT (spot) | spot | 94.2% | 0.117 | DODOUSDT | 63.05 bp | 100.0% | 3 | 473.1K | 115.6% |
| 52 | BAN | BANUSDT (usdm-futures) | usdm-futures | 94.2% | 0.142 | BTCUSDT | 59.94 bp | 100.0% | 2 | 1.3M | 97.3% |
| 53 | QI | QIUSDT (spot) | spot | 94.3% | 0.098 | BTCUSDT | 53.58 bp | 100.0% | 10 | 121.9K | 97.2% |
| 54 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 93.8% | 0.156 | BTCUSDT | 52.61 bp | 100.0% | 2 | 391.6K | 76.2% |
| 55 | QUICK | QUICKUSDT (spot) | spot | 94.5% | 0.089 | SKYAIUSDT | 51.31 bp | 100.0% | 2 | 54.6K | 74.9% |
| 56 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 93.8% | 0.121 | PYRUSDT | 26.70 bp | 100.0% | 1 | 2.2M | 58.2% |
| 57 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 93.6% | 0.147 | ZEREBROUSDT | 24.68 bp | 100.0% | 1 | 1.7M | 66.8% |
| 58 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 93.8% | 0.126 | ATMUSDT | 14.60 bp | 100.0% | 1 | 558.8K | 30.9% |
| 59 | KGST | KGSTUSDT (spot) | spot | 95.8% | 0.125 | TAGUSDT | 2.35 bp | 100.0% | 8 | 343.6K | 6.3% |
| 60 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 91.0% | 0.212 | DODOUSDT | 116.04 bp | 100.0% | 2 | 1.7M | 230.8% |
| 61 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 90.8% | 0.191 | PYRUSDT | 122.20 bp | 100.0% | 1 | 8.1M | 183.8% |
| 62 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 91.2% | 0.114 | HOMEUSDT | 113.60 bp | 100.0% | 1 | 2.6M | 164.3% |
| 63 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 90.6% | 0.139 | LABUSDT | 125.42 bp | 100.0% | 0 | 13.9M | 157.7% |
| 64 | BAS | BASUSDT (usdm-futures) | usdm-futures | 90.3% | 0.240 | LABUSDT | 119.64 bp | 100.0% | 1 | 3.2M | 170.9% |
| 65 | M | MUSDT (usdm-futures) | usdm-futures | 91.0% | 0.122 | ONUSDT | 100.16 bp | 100.0% | 1 | 3.2M | 165.0% |
| 66 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 90.9% | 0.144 | BUSDT | 97.93 bp | 100.0% | 1 | 1.2M | 168.8% |
| 67 | AIN | AINUSDT (usdm-futures) | usdm-futures | 90.2% | 0.162 | TAGUSDT | 92.15 bp | 100.0% | 1 | 984.9K | 133.5% |
| 68 | MMT | MMTUSDT (spot) | spot, usdm-futures | 89.7% | 0.157 | EVAAUSDT | 100.40 bp | 100.0% | 1 | 1.2M | 134.6% |
| 69 | KITE | KITEUSDT (spot) | spot, usdm-futures | 89.7% | 0.154 | BUSDT | 88.37 bp | 100.0% | 1 | 11.5M | 112.0% |
| 70 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 90.3% | 0.147 | BLUAIUSDT | 88.09 bp | 100.0% | 4 | 1.3M | 181.2% |
| 71 | ONE | ONEUSDT (spot) | spot, usdm-futures | 90.0% | 0.215 | BTCUSDT | 87.85 bp | 100.0% | 5 | 171.3K | 191.9% |
| 72 | OGN | OGNUSDT (spot) | spot, usdm-futures | 90.8% | 0.138 | BTCUSDT | 75.37 bp | 100.0% | 1 | 1M | 163.3% |
| 73 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 91.1% | 0.132 | ESPORTSUSDT | 67.31 bp | 100.0% | 2 | 1.2M | 91.1% |
| 74 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 89.7% | 0.313 | BTCUSDT | 63.33 bp | 100.0% | 2 | 880.8K | 127.1% |
| 75 | THE | THEUSDT (spot) | spot, usdm-futures | 89.3% | 0.154 | BASEDUSDT | 79.60 bp | 100.0% | 2 | 591.8K | 145.1% |
| 76 | VANA | VANAUSDT (spot) | spot, usdm-futures | 89.4% | 0.162 | BTCUSDT | 59.55 bp | 100.0% | 2 | 1.1M | 81.0% |
| 77 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 89.1% | 0.281 | THEUSDT | 54.48 bp | 100.0% | 1 | 832.6K | 81.4% |
| 78 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 91.1% | 0.132 | BTCUSDT | 53.40 bp | 100.0% | 3 | 801.5K | 125.1% |
| 79 | T | TUSDT (spot) | spot, usdm-futures | 88.7% | 0.199 | BUSDT | 107.60 bp | 100.0% | 4 | 1.7M | 198.9% |
| 80 | AWE | AWEUSDT (spot) | spot, usdm-futures | 88.4% | 0.149 | CLOUSDT | 67.11 bp | 100.0% | 1 | 346.5K | 101.9% |
| 81 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 88.4% | 0.144 | BTCUSDT | 66.10 bp | 100.0% | 1 | 1.1M | 90.1% |
| 82 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 89.4% | 0.150 | TAGUSDT | 48.30 bp | 100.0% | 1 | 2.3M | 74.5% |
| 83 | AMP | AMPUSDT (spot) | spot | 90.5% | 0.197 | KAITOUSDT | 42.54 bp | 100.0% | 5 | 470.5K | 113.2% |
| 84 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 87.7% | 0.123 | EVAAUSDT | 97.91 bp | 100.0% | 1 | 2.2M | 137.2% |
| 85 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 87.2% | 0.196 | BEATUSDT | 99.77 bp | 100.0% | 2 | 624.7K | 176.7% |
| 86 | SENT | SENTUSDT (spot) | spot, usdm-futures | 87.4% | 0.241 | KGSTUSDT | 85.77 bp | 100.0% | 1 | 1.8M | 131.4% |
| 87 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 87.8% | 0.143 | EVAAUSDT | 79.88 bp | 100.0% | 1 | 2.8M | 127.0% |
| 88 | LA | LAUSDT (spot) | spot, usdm-futures | 87.7% | 0.181 | BANKUSDT | 76.53 bp | 100.0% | 2 | 495.5K | 131.6% |
| 89 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 87.1% | 0.154 | RIFUSDT | 66.12 bp | 100.0% | 1 | 3.6M | 103.4% |
| 90 | PIVX | PIVXUSDT (spot) | spot | 86.6% | 0.257 | BLESSUSDT | 100.18 bp | 100.0% | 2 | 298.5K | 144.6% |
| 91 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 87.0% | 0.144 | HANAUSDT | 56.42 bp | 100.0% | 2 | 612.4K | 72.8% |
| 92 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 86.0% | 0.159 | TREEUSDT | 88.69 bp | 100.0% | 1 | 2M | 129.7% |
| 93 | TUT | TUTUSDT (spot) | spot, usdm-futures | 85.6% | 0.169 | BANKUSDT | 86.29 bp | 100.0% | 1 | 471K | 112.6% |
| 94 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 86.2% | 0.149 | ACEUSDT | 85.69 bp | 100.0% | 1 | 2.5M | 122.8% |
| 95 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 85.0% | 0.270 | TAGUSDT | 161.97 bp | 100.0% | 1 | 21.1M | 282.4% |
| 96 | CATI | CATIUSDT (spot) | spot, usdm-futures | 85.8% | 0.173 | BTCUSDT | 79.97 bp | 100.0% | 2 | 340.5K | 101.3% |
| 97 | BR | BRUSDT (usdm-futures) | usdm-futures | 85.3% | 0.172 | QUICKUSDT | 78.78 bp | 100.0% | 1 | 945.2K | 109.9% |
| 98 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 84.9% | 0.174 | EVAAUSDT | 74.83 bp | 100.0% | 2 | 858.9K | 117.5% |
| 99 | AT | ATUSDT (spot) | spot, usdm-futures | 85.6% | 0.179 | SENTUSDT | 58.87 bp | 100.0% | 1 | 288.6K | 83.7% |
| 100 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 85.6% | 0.194 | BTCUSDT | 54.73 bp | 100.0% | 2 | 1M | 77.9% |
| 101 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 85.0% | 0.291 | BTCUSDT | 49.26 bp | 100.0% | 5 | 664K | 64.9% |
| 102 | GNO | GNOUSDT (spot) | spot | 85.8% | 0.208 | AKEUSDT | 41.19 bp | 100.0% | 2 | 124.9K | 72.7% |
| 103 | O | OUSDT (usdm-futures) | usdm-futures | 83.5% | 0.148 | KITEUSDT | 95.88 bp | 100.0% | 2 | 9.1M | 145.8% |
| 104 | HOT | HOTUSDT (spot) | spot, usdm-futures | 83.4% | 0.286 | BTCUSDT | 70.57 bp | 100.0% | 3 | 256K | 108.1% |
| 105 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 83.2% | 0.278 | BLESSUSDT | 85.32 bp | 100.0% | 1 | 1M | 136.5% |
| 106 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 83.4% | 0.284 | BTCUSDT | 70.46 bp | 100.0% | 1 | 1.3M | 89.7% |
| 107 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 83.4% | 0.263 | BTCUSDT | 62.62 bp | 100.0% | 2 | 329.9K | 84.8% |
| 108 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 83.0% | 0.171 | HOMEUSDT | 49.05 bp | 100.0% | 1 | 1.7M | 87.8% |
| 109 | GRAM | GRAMUSDT (spot) | spot, usdm-futures | 82.7% | 0.335 | BTCUSDT | 47.18 bp | 100.0% | 2 | 7.4M | 66.4% |
| 110 | TA | TAUSDT (usdm-futures) | usdm-futures | 83.5% | 0.158 | SYNUSDT | 46.74 bp | 100.0% | 1 | 1.1M | 66.2% |
| 111 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 82.0% | 0.206 | BASUSDT | 187.25 bp | 100.0% | 1 | 23.9M | 259.3% |
| 112 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 81.5% | 0.215 | LABUSDT | 145.25 bp | 100.0% | 1 | 10.5M | 194.1% |
| 113 | GLMR | GLMRUSDT (spot) | spot | 81.6% | 0.249 | POWERUSDT | 68.74 bp | 100.0% | 9 | 409.3K | 100.9% |
| 114 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 81.4% | 0.219 | LABUSDT | 61.75 bp | 100.0% | 1 | 452K | 78.1% |
| 115 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 81.1% | 0.285 | BTCUSDT | 61.93 bp | 100.0% | 1 | 1.5M | 91.1% |
| 116 | Q | QUSDT (usdm-futures) | usdm-futures | 80.8% | 0.158 | OUSDT | 80.55 bp | 100.0% | 1 | 1.7M | 115.1% |
| 117 | OPN | OPNUSDT (spot) | spot, usdm-futures | 80.5% | 0.212 | MANTAUSDT | 79.52 bp | 100.0% | 3 | 15.1M | 112.2% |
| 118 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 80.4% | 0.207 | BTCUSDT | 71.26 bp | 100.0% | 2 | 768.7K | 102.2% |
| 119 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 80.1% | 0.181 | SAPIENUSDT | 101.25 bp | 100.0% | 1 | 7.1M | 134.5% |
| 120 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 80.0% | 0.206 | LABUSDT | 75.23 bp | 100.0% | 2 | 558.8K | 102.8% |
| 121 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 79.9% | 0.260 | VANAUSDT | 72.67 bp | 100.0% | 4 | 618.9K | 96.8% |
| 122 | AI | AIUSDT (spot) | spot | 80.0% | 0.155 | ENSOUSDT | 61.52 bp | 100.0% | 5 | 342.4K | 110.1% |
| 123 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 79.4% | 0.246 | LABUSDT | 90.00 bp | 100.0% | 1 | 44.4M | 123.8% |
| 124 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 79.2% | 0.181 | QIUSDT | 78.36 bp | 100.0% | 1 | 2.2M | 107.6% |
| 125 | FTT | FTTUSDT (spot) | spot | 79.4% | 0.343 | BTCUSDT | 56.24 bp | 100.0% | 2 | 190K | 84.5% |
| 126 | JST | JSTUSDT (spot) | spot, usdm-futures | 80.2% | 0.163 | BILLUSDT | 35.11 bp | 100.0% | 0 | 3.2M | 45.2% |
| 127 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 78.9% | 0.193 | BTCUSDT | 30.74 bp | 100.0% | 3 | 2.5M | 38.9% |
| 128 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 78.5% | 0.285 | BANKUSDT | 223.19 bp | 100.0% | 1 | 22.3M | 452.5% |
| 129 | H | HUSDT (usdm-futures) | usdm-futures | 78.3% | 0.205 | LABUSDT | 85.76 bp | 100.0% | 1 | 6.7M | 121.5% |
| 130 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 77.7% | 0.189 | PROMUSDT | 82.26 bp | 100.0% | 2 | 1M | 145.3% |
| 131 | BEL | BELUSDT (spot) | spot, usdm-futures | 77.8% | 0.161 | HEIUSDT | 79.04 bp | 100.0% | 1 | 741.2K | 98.8% |
| 132 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 77.5% | 0.185 | 币安人生USDT | 40.78 bp | 100.0% | 1 | 644K | 54.5% |
| 133 | WIN | WINUSDT (spot) | spot | 77.9% | 0.330 | BTCUSDT | 38.63 bp | 100.0% | 2 | 97.9K | 51.7% |
| 134 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 79.3% | 0.168 | PROMUSDT | 25.05 bp | 100.0% | 1 | 260.5K | 44.2% |
| 135 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 77.1% | 0.233 | KAITOUSDT | 85.02 bp | 100.0% | 1 | 1.2M | 114.2% |
| 136 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 76.9% | 0.212 | BTCUSDT | 70.01 bp | 100.0% | 2 | 1.7M | 104.6% |
| 137 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 76.1% | 0.229 | SKYAIUSDT | 129.22 bp | 100.0% | 1 | 5.9M | 185.4% |
| 138 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 76.1% | 0.301 | BTCUSDT | 59.10 bp | 100.0% | 1 | 1.6M | 84.7% |
| 139 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 76.3% | 0.307 | TUSDT | 48.68 bp | 100.0% | 5 | 233.1K | 90.2% |
| 140 | BX | BXUSDT (usdm-futures) | usdm-futures | 76.4% | 0.162 | BTCUSDT | 33.19 bp | 100.0% | 1 | 804.9K | 47.8% |
| 141 | ASTS | ASTSUSDT (usdm-futures) | usdm-futures | 75.4% | 0.220 | BTCUSDT | 66.57 bp | 100.0% | 1 | 2.5M | 120.2% |
| 142 | KNC | KNCUSDT (spot) | spot, usdm-futures | 75.8% | 0.247 | TAUSDT | 36.53 bp | 100.0% | 3 | 65.5K | 76.0% |
| 143 | APR | APRUSDT (usdm-futures) | usdm-futures | 75.3% | 0.286 | AIOTUSDT | 75.12 bp | 100.0% | 2 | 3.3M | 105.2% |
| 144 | ID | IDUSDT (spot) | spot, usdm-futures | 75.0% | 0.244 | CHIPUSDT | 68.92 bp | 100.0% | 2 | 702.3K | 89.4% |
| 145 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 75.3% | 0.201 | PROMUSDT | 66.84 bp | 100.0% | 1 | 2M | 91.5% |
| 146 | SKY | SKYUSDT (spot) | spot, usdm-futures | 74.6% | 0.290 | BTCUSDT | 64.93 bp | 100.0% | 1 | 1.2M | 80.0% |
| 147 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 74.6% | 0.273 | BTCUSDT | 47.59 bp | 100.0% | 2 | 167.4K | 64.8% |
| 148 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 75.7% | 0.198 | ASTSUSDT | 23.90 bp | 100.0% | 2 | 457.5K | 41.4% |
| 149 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 73.8% | 0.263 | BLESSUSDT | 70.84 bp | 100.0% | 3 | 2.2M | 95.0% |
| 150 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 74.1% | 0.205 | SLXUSDT | 61.19 bp | 100.0% | 2 | 1.4M | 82.5% |
| 151 | A | AUSDT (spot) | spot, usdm-futures | 74.1% | 0.219 | BTCUSDT | 58.10 bp | 100.0% | 2 | 385.1K | 78.4% |
| 152 | STRAX | STRAXUSDT (spot) | spot | 73.4% | 0.261 | MAGMAUSDT | 53.16 bp | 100.0% | 2 | 355.8K | 116.5% |
| 153 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 73.5% | 0.230 | LAUSDT | 48.05 bp | 100.0% | 1 | 587K | 66.7% |
| 154 | GNS | GNSUSDT (spot) | spot | 73.2% | 0.252 | ACEUSDT | 27.40 bp | 100.0% | 5 | 50.7K | 41.7% |
| 155 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 74.3% | 0.167 | AIUSDT | 21.16 bp | 100.0% | 4 | 185.3K | 30.1% |
| 156 | U | UUSDT (spot) | spot | 74.8% | 0.149 | B2USDT | 0.54 bp | 100.0% | 7 | 14.9M | 0.7% |
| 157 | STAR | STARUSDT (usdm-futures) | usdm-futures | 71.7% | 0.192 | ROBOUSDT | 138.31 bp | 100.0% | 0 | 2.5M | 213.3% |
| 158 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 71.7% | 0.174 | TUTUSDT | 104.44 bp | 100.0% | 3 | 3.4M | 147.0% |
| 159 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 70.3% | 0.139 | ONUSDT | 121.96 bp | 100.0% | 1 | 2.9M | 152.4% |
| 160 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 70.3% | 0.255 | 币安人生USDT | 86.69 bp | 100.0% | 1 | 3.5M | 142.0% |
| 161 | JTO | JTOUSDT (spot) | spot, usdm-futures | 70.9% | 0.203 | PIVXUSDT | 84.28 bp | 100.0% | 1 | 3.1M | 115.5% |
| 162 | YB | YBUSDT (spot) | spot, usdm-futures | 70.8% | 0.238 | BLUAIUSDT | 75.85 bp | 100.0% | 2 | 328.2K | 116.8% |
| 163 | KAT | KATUSDT (spot) | spot, usdm-futures | 69.9% | 0.222 | VELVETUSDT | 75.90 bp | 100.0% | 2 | 818.2K | 117.0% |
| 164 | TST | TSTUSDT (spot) | spot, usdm-futures | 70.0% | 0.197 | BTCUSDT | 66.60 bp | 100.0% | 2 | 570.5K | 85.5% |
| 165 | PHA | PHAUSDT (spot) | spot, usdm-futures | 69.9% | 0.359 | BTCUSDT | 64.55 bp | 100.0% | 4 | 477.3K | 86.0% |
| 166 | GPS | GPSUSDT (spot) | spot, usdm-futures | 71.5% | 0.185 | BTCUSDT | 64.48 bp | 100.0% | 2 | 375.2K | 90.6% |
| 167 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 69.6% | 0.255 | PORTOUSDT | 62.06 bp | 100.0% | 1 | 1.7M | 90.3% |
| 168 | 4 | 4USDT (usdm-futures) | usdm-futures | 70.5% | 0.183 | ANTHROPICUSDT | 57.06 bp | 100.0% | 1 | 1.8M | 74.0% |
| 169 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 69.3% | 0.253 | SENTUSDT | 75.13 bp | 100.0% | 1 | 1.1M | 102.1% |
| 170 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 69.7% | 0.307 | BTCUSDT | 52.04 bp | 100.0% | 4 | 1.9M | 80.3% |
| 171 | CC | CCUSDT (usdm-futures) | usdm-futures | 70.7% | 0.160 | DATAIPUSDT | 44.93 bp | 100.0% | 1 | 3.7M | 58.6% |
| 172 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 68.6% | 0.236 | PRLUSDT | 178.55 bp | 100.0% | 1 | 13.6M | 243.7% |
| 173 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 67.9% | 0.216 | ROBOUSDT | 95.09 bp | 100.0% | 0 | 2.6M | 139.4% |
| 174 | MITO | MITOUSDT (spot) | spot, usdm-futures | 68.4% | 0.289 | THEUSDT | 74.07 bp | 100.0% | 2 | 634.3K | 107.3% |
| 175 | SOLV | SOLVUSDT (spot) | spot, usdm-futures | 67.8% | 0.350 | ARIAUSDT | 60.90 bp | 100.0% | 6 | 187.1K | 99.2% |
| 176 | 2Z | 2ZUSDT (spot) | spot, usdm-futures | 68.2% | 0.317 | BTCUSDT | 50.48 bp | 100.0% | 1 | 200.1K | 65.9% |
| 177 | FF | FFUSDT (spot) | spot, usdm-futures | 68.3% | 0.236 | AKEUSDT | 48.43 bp | 100.0% | 1 | 1.2M | 77.6% |
| 178 | MBL | MBLUSDT (spot) | spot | 69.3% | 0.248 | BTCUSDT | 39.92 bp | 100.0% | 3 | 450.3K | 51.7% |
| 179 | HFT | HFTUSDT (spot) | spot, usdm-futures | 67.4% | 0.341 | BTCUSDT | 77.70 bp | 100.0% | 5 | 130.8K | 102.8% |
| 180 | MAV | MAVUSDT (spot) | spot, usdm-futures | 66.7% | 0.325 | BTCUSDT | 54.09 bp | 100.0% | 2 | 179.9K | 90.0% |
| 181 | NMR | NMRUSDT (spot) | spot, usdm-futures | 66.7% | 0.221 | BTCUSDT | 45.16 bp | 100.0% | 2 | 220.2K | 58.8% |
| 182 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 68.3% | 0.379 | BTCUSDT | 37.82 bp | 100.0% | 0 | 23M | 45.8% |
| 183 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 68.3% | 0.206 | CSCOUSDT | 22.97 bp | 100.0% | 2 | 212.8K | 33.8% |
| 184 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 66.4% | 0.283 | KAITOUSDT | 23.19 bp | 100.0% | 10 | 189.7K | 37.6% |
| 185 | STRC | STRCUSDT (usdm-futures) | usdm-futures | 67.9% | 0.267 | BTCUSDT | 17.73 bp | 100.0% | 3 | 1.1M | 25.7% |
| 186 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 64.5% | 0.381 | TLMUSDT | 177.13 bp | 100.0% | 1 | 5.3M | 245.1% |
| 187 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 65.5% | 0.264 | ACEUSDT | 128.08 bp | 100.0% | 1 | 1.6M | 199.4% |
| 188 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 64.5% | 0.231 | SKYAIUSDT | 110.04 bp | 100.0% | 1 | 2.6M | 146.5% |
| 189 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 65.3% | 0.248 | KAITOUSDT | 100.38 bp | 100.0% | 1 | 1.9M | 142.3% |
| 190 | ERA | ERAUSDT (spot) | spot, usdm-futures | 63.6% | 0.420 | LAUSDT | 98.53 bp | 100.0% | 3 | 486.2K | 233.4% |
| 191 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 64.4% | 0.447 | BLESSUSDT | 92.02 bp | 100.0% | 1 | 691.1K | 191.1% |
| 192 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 63.8% | 0.248 | BTCUSDT | 86.04 bp | 100.0% | 1 | 1.5M | 115.6% |
| 193 | DCR | DCRUSDT (spot) | spot | 65.0% | 0.281 | LUMIAUSDT | 83.00 bp | 100.0% | 2 | 250.6K | 164.0% |
| 194 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 62.9% | 0.183 | AGLDUSDT | 89.72 bp | 100.0% | 3 | 896.4K | 112.3% |
| 195 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 63.3% | 0.190 | JELLYJELLYUSDT | 81.90 bp | 100.0% | 0 | 1.5M | 106.8% |
| 196 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 63.0% | 0.278 | PHAUSDT | 79.97 bp | 100.0% | 1 | 961.2K | 108.1% |
| 197 | LDO | LDOUSDT (spot) | spot, usdm-futures | 63.2% | 0.339 | BTCUSDT | 79.65 bp | 100.0% | 1 | 3.5M | 101.7% |
| 198 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 63.0% | 0.238 | ARXUSDT | 70.16 bp | 100.0% | 1 | 2.9M | 95.0% |
| 199 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 64.0% | 0.266 | MUSDT | 67.86 bp | 100.0% | 2 | 1.5M | 91.9% |
| 200 | ZRO | ZROUSDT (spot) | spot, usdm-futures | 62.4% | 0.280 | BTCUSDT | 63.84 bp | 100.0% | 1 | 1.9M | 80.4% |
| 201 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 62.0% | 0.298 | MITOUSDT | 115.17 bp | 100.0% | 1 | 5.3M | 147.9% |
| 202 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 62.0% | 0.282 | BTCUSDT | 56.75 bp | 100.0% | 2 | 4.5M | 84.2% |
| 203 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 61.6% | 0.253 | KNCUSDT | 63.35 bp | 100.0% | 1 | 416.6K | 82.0% |
| 204 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 63.1% | 0.289 | TOSHIUSDT | 52.85 bp | 100.0% | 1 | 150.7K | 72.3% |
| 205 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 61.5% | 0.284 | BTCUSDT | 65.92 bp | 100.0% | 1 | 2.1M | 85.7% |
| 206 | RIVN | RIVNUSDT (usdm-futures) | usdm-futures | 61.4% | 0.300 | BXUSDT | 43.33 bp | 100.0% | 3 | 393.4K | 71.4% |
| 207 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 62.8% | 0.373 | ASTSUSDT | 40.27 bp | 100.0% | 1 | 19.2M | 63.8% |
| 208 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 60.6% | 0.268 | BTCUSDT | 63.46 bp | 100.0% | 2 | 1M | 97.3% |
| 209 | ESP | ESPUSDT (spot) | spot, usdm-futures | 60.4% | 0.193 | PYRUSDT | 49.78 bp | 100.0% | 1 | 382K | 66.6% |
| 210 | BTR | BTRUSDT (usdm-futures) | usdm-futures | 60.5% | 0.392 | BTCUSDT | 48.00 bp | 100.0% | 2 | 736.4K | 66.0% |
| 211 | AUDIO | AUDIOUSDT (spot) | spot | 61.8% | 0.205 | AMPUSDT | 39.47 bp | 100.0% | 2 | 373.7K | 75.9% |
| 212 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 59.5% | 0.303 | LABUSDT | 102.24 bp | 100.0% | 2 | 7.8M | 138.4% |
| 213 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 59.4% | 0.373 | BTCUSDT | 65.41 bp | 100.0% | 1 | 3.1M | 84.5% |
| 214 | BANANA | BANANAUSDT (spot) | spot, usdm-futures | 59.4% | 0.312 | BTCUSDT | 57.21 bp | 100.0% | 1 | 378.7K | 75.2% |
| 215 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 59.3% | 0.236 | RIVNUSDT | 40.80 bp | 100.0% | 6 | 419.6K | 66.7% |
| 216 | TFUEL | TFUELUSDT (spot) | spot | 59.3% | 0.323 | BTCUSDT | 40.19 bp | 100.0% | 2 | 98.2K | 50.1% |
| 217 | SUN | SUNUSDT (spot) | spot, usdm-futures | 60.4% | 0.196 | THEUSDT | 17.68 bp | 100.0% | 3 | 644.6K | 24.6% |
| 218 | PUMP | PUMPUSDT (spot) | spot, usdm-futures | 57.1% | 0.346 | BTCUSDT | 88.14 bp | 100.0% | 1 | 9.1M | 119.6% |
| 219 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 56.9% | 0.259 | DGBUSDT | 87.82 bp | 100.0% | 1 | 3.7M | 146.6% |
| 220 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 57.8% | 0.207 | ROBOUSDT | 77.83 bp | 100.0% | 1 | 3M | 98.9% |
| 221 | WET | WETUSDT (usdm-futures) | usdm-futures | 57.7% | 0.267 | BTCUSDT | 71.49 bp | 100.0% | 1 | 1.5M | 94.0% |

## Diagnostics

- Basis size selected: 221
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.055
- Maximum pairwise absolute correlation: 0.447
- Mean whole-market projection R²: 84.2%
- Median whole-market projection R²: 80.0%
- 10th-percentile whole-market projection R²: 70.9%
- Minimum whole-market projection R²: 66.0%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.4% | 10.8% | 0.5% | 0.0% |
| 5 | 17.7% | 13.0% | 1.9% | 0.1% |
| 10 | 19.8% | 15.3% | 3.4% | 0.5% |
| 15 | 21.7% | 16.5% | 5.1% | 1.9% |
| 20 | 23.7% | 18.3% | 6.3% | 2.6% |
| 25 | 25.6% | 19.9% | 8.1% | 3.3% |
| 30 | 27.5% | 21.1% | 9.2% | 4.7% |
| 35 | 29.4% | 22.8% | 10.9% | 5.2% |
| 40 | 31.7% | 25.0% | 12.5% | 6.0% |
| 45 | 33.7% | 27.0% | 14.1% | 7.2% |
| 50 | 35.7% | 28.7% | 15.6% | 7.4% |
| 55 | 37.7% | 30.6% | 17.4% | 7.6% |
| 60 | 39.5% | 32.6% | 19.3% | 13.3% |
| 65 | 41.3% | 34.3% | 21.3% | 14.7% |
| 70 | 43.0% | 35.7% | 22.8% | 15.6% |
| 75 | 44.9% | 37.2% | 24.4% | 16.1% |
| 80 | 46.7% | 38.6% | 25.9% | 18.1% |
| 85 | 48.4% | 40.1% | 27.5% | 20.0% |
| 90 | 50.5% | 42.7% | 29.6% | 21.0% |
| 95 | 52.1% | 44.2% | 31.4% | 24.1% |
| 100 | 53.6% | 46.0% | 32.8% | 24.6% |
| 105 | 55.1% | 47.5% | 34.3% | 27.3% |
| 110 | 56.7% | 49.2% | 36.1% | 29.4% |
| 115 | 58.4% | 50.8% | 38.1% | 31.4% |
| 120 | 59.9% | 52.5% | 39.5% | 32.9% |
| 125 | 61.6% | 54.1% | 41.6% | 34.7% |
| 130 | 63.1% | 56.0% | 43.0% | 37.1% |
| 135 | 64.6% | 57.3% | 44.8% | 38.2% |
| 140 | 66.0% | 58.9% | 46.6% | 40.1% |
| 145 | 67.4% | 60.7% | 48.4% | 41.5% |
| 150 | 69.1% | 63.1% | 50.2% | 43.4% |
| 155 | 70.3% | 64.7% | 51.6% | 44.0% |
| 160 | 71.5% | 65.8% | 52.9% | 48.0% |
| 165 | 72.7% | 67.1% | 54.4% | 48.9% |
| 170 | 73.9% | 68.4% | 55.9% | 50.0% |
| 175 | 75.0% | 69.4% | 57.6% | 51.9% |
| 180 | 76.1% | 70.5% | 58.9% | 53.3% |
| 185 | 77.1% | 71.5% | 60.6% | 56.5% |
| 190 | 78.1% | 72.7% | 62.0% | 57.5% |
| 195 | 79.1% | 74.0% | 63.2% | 58.7% |
| 200 | 80.2% | 75.2% | 64.8% | 59.7% |
| 205 | 81.1% | 76.4% | 66.2% | 60.5% |
| 210 | 82.1% | 77.6% | 67.3% | 61.8% |
| 215 | 83.1% | 78.9% | 69.3% | 63.2% |
| 220 | 84.0% | 79.7% | 70.6% | 66.0% |
| 221 | 84.2% | 80.0% | 70.9% | 66.0% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | LAB | EVAA | ESPORTS | BANK | AKE | B | TAG | US | TLM | VELVET | SYN | CLO | DODO | ON | SKYAI | RIF | MAGMA | AVAAI | GWEI | PYR | BTTC | HOME | ALLO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.160 | 0.072 | 0.103 | 0.033 | 0.027 | 0.008 | 0.089 | 0.068 | -0.047 | 0.008 | -0.021 | 0.004 | 0.028 | 0.007 | 0.023 | -0.026 | 0.022 | 0.064 | 0.122 | 0.076 | 0.067 | 0.114 | 0.073 |
| LAB | 0.160 | 1.000 | 0.073 | 0.083 | 0.060 | 0.048 | -0.033 | 0.162 | 0.033 | 0.043 | 0.099 | 0.107 | 0.056 | 0.032 | 0.090 | 0.104 | -0.031 | 0.067 | -0.021 | 0.111 | 0.048 | 0.022 | 0.084 | 0.027 |
| EVAA | 0.072 | 0.073 | 1.000 | 0.079 | 0.021 | 0.031 | 0.113 | -0.044 | 0.023 | -0.004 | 0.023 | 0.033 | 0.041 | 0.005 | 0.064 | 0.076 | 0.064 | -0.068 | 0.032 | 0.024 | 0.004 | -0.008 | -0.012 | -0.005 |
| ESPORTS | 0.103 | 0.083 | 0.079 | 1.000 | 0.069 | 0.044 | 0.061 | 0.020 | 0.114 | -0.003 | 0.031 | 0.004 | -0.034 | 0.053 | 0.043 | 0.079 | -0.063 | 0.058 | -0.157 | 0.088 | -0.010 | -0.030 | -0.022 | -0.053 |
| BANK | 0.033 | 0.060 | 0.021 | 0.069 | 1.000 | -0.028 | 0.070 | 0.044 | 0.059 | 0.080 | 0.089 | 0.079 | 0.042 | -0.014 | 0.027 | 0.057 | 0.043 | -0.031 | 0.036 | 0.018 | -0.032 | 0.007 | 0.079 | -0.006 |
| AKE | 0.027 | 0.048 | 0.031 | 0.044 | -0.028 | 1.000 | 0.015 | 0.016 | 0.094 | -0.002 | -0.038 | -0.028 | 0.045 | -0.019 | -0.039 | -0.058 | 0.019 | 0.002 | -0.002 | -0.050 | 0.006 | -0.031 | -0.098 | -0.022 |
| B | 0.008 | -0.033 | 0.113 | 0.061 | 0.070 | 0.015 | 1.000 | 0.032 | 0.052 | 0.032 | -0.010 | -0.049 | 0.118 | -0.012 | 0.003 | 0.045 | 0.052 | -0.020 | -0.032 | 0.004 | -0.193 | 0.020 | -0.006 | 0.034 |
| TAG | 0.089 | 0.162 | -0.044 | 0.020 | 0.044 | 0.016 | 0.032 | 1.000 | -0.015 | 0.036 | 0.116 | -0.024 | -0.012 | 0.007 | 0.024 | 0.079 | -0.015 | -0.065 | -0.007 | -0.048 | -0.042 | 0.022 | 0.094 | 0.074 |
| US | 0.068 | 0.033 | 0.023 | 0.114 | 0.059 | 0.094 | 0.052 | -0.015 | 1.000 | -0.068 | -0.011 | -0.025 | -0.006 | 0.015 | 0.040 | 0.070 | 0.005 | 0.062 | 0.055 | -0.029 | 0.072 | 0.045 | -0.062 | 0.042 |
| TLM | -0.047 | 0.043 | -0.004 | -0.003 | 0.080 | -0.002 | 0.032 | 0.036 | -0.068 | 1.000 | 0.093 | 0.079 | 0.124 | 0.003 | 0.067 | 0.002 | 0.108 | -0.042 | -0.117 | -0.012 | 0.043 | -0.054 | -0.026 | 0.015 |
| VELVET | 0.008 | 0.099 | 0.023 | 0.031 | 0.089 | -0.038 | -0.010 | 0.116 | -0.011 | 0.093 | 1.000 | 0.040 | 0.021 | -0.007 | 0.055 | 0.101 | -0.042 | 0.088 | 0.017 | 0.031 | -0.001 | -0.040 | -0.023 | 0.012 |
| SYN | -0.021 | 0.107 | 0.033 | 0.004 | 0.079 | -0.028 | -0.049 | -0.024 | -0.025 | 0.079 | 0.040 | 1.000 | 0.042 | -0.009 | 0.028 | -0.025 | 0.042 | 0.051 | 0.045 | -0.013 | 0.024 | 0.003 | -0.025 | 0.004 |
| CLO | 0.004 | 0.056 | 0.041 | -0.034 | 0.042 | 0.045 | 0.118 | -0.012 | -0.006 | 0.124 | 0.021 | 0.042 | 1.000 | 0.010 | 0.053 | -0.064 | 0.005 | 0.022 | 0.060 | 0.014 | 0.031 | -0.101 | -0.010 | -0.080 |
| DODO | 0.028 | 0.032 | 0.005 | 0.053 | -0.014 | -0.019 | -0.012 | 0.007 | 0.015 | 0.003 | -0.007 | -0.009 | 0.010 | 1.000 | 0.016 | 0.021 | -0.086 | 0.028 | -0.072 | 0.052 | 0.011 | -0.018 | -0.035 | -0.007 |
| ON | 0.007 | 0.090 | 0.064 | 0.043 | 0.027 | -0.039 | 0.003 | 0.024 | 0.040 | 0.067 | 0.055 | 0.028 | 0.053 | 0.016 | 1.000 | -0.005 | 0.016 | 0.008 | 0.012 | 0.017 | -0.023 | -0.017 | -0.028 | -0.022 |
| SKYAI | 0.023 | 0.104 | 0.076 | 0.079 | 0.057 | -0.058 | 0.045 | 0.079 | 0.070 | 0.002 | 0.101 | -0.025 | -0.064 | 0.021 | -0.005 | 1.000 | 0.029 | -0.023 | 0.024 | -0.011 | 0.009 | -0.009 | 0.006 | -0.023 |
| RIF | -0.026 | -0.031 | 0.064 | -0.063 | 0.043 | 0.019 | 0.052 | -0.015 | 0.005 | 0.108 | -0.042 | 0.042 | 0.005 | -0.086 | 0.016 | 0.029 | 1.000 | 0.040 | 0.055 | -0.062 | 0.018 | 0.019 | 0.031 | 0.007 |
| MAGMA | 0.022 | 0.067 | -0.068 | 0.058 | -0.031 | 0.002 | -0.020 | -0.065 | 0.062 | -0.042 | 0.088 | 0.051 | 0.022 | 0.028 | 0.008 | -0.023 | 0.040 | 1.000 | 0.009 | 0.001 | 0.021 | -0.045 | -0.011 | 0.001 |
| AVAAI | 0.064 | -0.021 | 0.032 | -0.157 | 0.036 | -0.002 | -0.032 | -0.007 | 0.055 | -0.117 | 0.017 | 0.045 | 0.060 | -0.072 | 0.012 | 0.024 | 0.055 | 0.009 | 1.000 | 0.003 | -0.062 | -0.016 | -0.053 | -0.029 |
| GWEI | 0.122 | 0.111 | 0.024 | 0.088 | 0.018 | -0.050 | 0.004 | -0.048 | -0.029 | -0.012 | 0.031 | -0.013 | 0.014 | 0.052 | 0.017 | -0.011 | -0.062 | 0.001 | 0.003 | 1.000 | 0.028 | -0.086 | 0.072 | 0.002 |
| PYR | 0.076 | 0.048 | 0.004 | -0.010 | -0.032 | 0.006 | -0.193 | -0.042 | 0.072 | 0.043 | -0.001 | 0.024 | 0.031 | 0.011 | -0.023 | 0.009 | 0.018 | 0.021 | -0.062 | 0.028 | 1.000 | 0.005 | 0.048 | -0.011 |
| BTTC | 0.067 | 0.022 | -0.008 | -0.030 | 0.007 | -0.031 | 0.020 | 0.022 | 0.045 | -0.054 | -0.040 | 0.003 | -0.101 | -0.018 | -0.017 | -0.009 | 0.019 | -0.045 | -0.016 | -0.086 | 0.005 | 1.000 | -0.010 | -0.007 |
| HOME | 0.114 | 0.084 | -0.012 | -0.022 | 0.079 | -0.098 | -0.006 | 0.094 | -0.062 | -0.026 | -0.023 | -0.025 | -0.010 | -0.035 | -0.028 | 0.006 | 0.031 | -0.011 | -0.053 | 0.072 | 0.048 | -0.010 | 1.000 | -0.007 |
| ALLO | 0.073 | 0.027 | -0.005 | -0.053 | -0.006 | -0.022 | 0.034 | 0.074 | 0.042 | 0.015 | 0.012 | 0.004 | -0.080 | -0.007 | -0.022 | -0.023 | 0.007 | 0.001 | -0.029 | 0.002 | -0.011 | -0.007 | -0.007 | 1.000 |

The complete 221 × 221 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| MAVIA | MAVIAUSDT | 66.0% | 58.3% | BTCUSDT | 0.353 |
| ZKP | ZKPUSDT | 66.6% | 57.8% | BTCUSDT | 0.293 |
| UMA | UMAUSDT | 66.6% | 57.8% | BTCUSDT | 0.389 |
| G | GUSDT | 67.1% | 57.4% | PHAUSDT | 0.292 |
| PYTH | PYTHUSDT | 67.1% | 57.4% | BTCUSDT | 0.260 |
| ACM | ACMUSDT | 67.2% | 57.3% | BARUSDT | 0.379 |
| QNT | QNTUSDT | 67.3% | 57.2% | BTCUSDT | 0.513 |
| STORJ | STORJUSDT | 67.4% | 57.1% | BTCUSDT | 0.386 |
| CRM | CRMUSDT | 67.5% | 57.0% | 币安人生USDT | 0.249 |
| SC | SCUSDT | 67.5% | 57.0% | BTCUSDT | 0.446 |
| RARE | RAREUSDT | 67.6% | 57.0% | PYRUSDT | -0.320 |
| KAVA | KAVAUSDT | 67.6% | 56.9% | KNCUSDT | 0.297 |
| POL | POLUSDT | 67.7% | 56.9% | TAKEUSDT | 0.250 |
| AAPL | AAPLUSDT | 67.8% | 56.8% | CSCOUSDT | -0.306 |
| C | CUSDT | 67.9% | 56.7% | BTCUSDT | 0.320 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

