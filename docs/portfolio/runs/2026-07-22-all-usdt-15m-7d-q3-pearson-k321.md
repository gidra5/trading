# Binance portfolio basis

Generated 2026-07-23T18:24:08.165Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-16T00:00Z through 2026-07-22T23:45Z
- Sampling: 15m log returns (672 samples over 7 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 691 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.2B | 30.9% |
| 2 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 100.0% | 0.000 | BTCUSDT | 100.0% | 0.3 | 31.2M | 556.9% |
| 3 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 100.0% | 0.010 | BTCUSDT | 100.0% | 0.3 | 13.4M | 289.9% |
| 4 | WEN | WENUSDT (usdm-futures) | usdm-futures | 100.0% | 0.010 | BULLAUSDT | 100.0% | 1.8 | 493.6K | 49.0% |
| 5 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 100.0% | 0.016 | DEXEUSDT | 100.0% | 1 | 862.5K | 180.3% |
| 6 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 99.9% | 0.030 | BTCUSDT | 100.0% | 0.5 | 2.1M | 90.4% |
| 7 | ONE | ONEUSDT (spot) | spot, usdm-futures | 99.9% | 0.034 | DEXEUSDT | 100.0% | 2.5 | 638K | 343.2% |
| 8 | QI | QIUSDT (spot) | spot | 99.8% | 0.049 | EPICUSDT | 100.0% | 3.5 | 136.8K | 114.5% |
| 9 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 99.7% | 0.060 | BTCUSDT | 100.0% | 0.3 | 4.5M | 268.7% |
| 10 | XNO | XNOUSDT (spot) | spot | 99.7% | 0.042 | EPICUSDT | 100.0% | 1.8 | 459.5K | 399.9% |
| 11 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 99.7% | 0.065 | BTCUSDT | 100.0% | 0.5 | 3.5M | 107.6% |
| 12 | TURTLE | TURTLEUSDT (spot) | spot, usdm-futures | 99.6% | 0.041 | QIUSDT | 100.0% | 4.3 | 143.7K | 103.9% |
| 13 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 99.5% | 0.060 | STABLEUSDT | 100.0% | 0.8 | 2.4M | 178.6% |
| 14 | AMP | AMPUSDT (spot) | spot | 99.5% | 0.061 | BTCUSDT | 100.0% | 1.5 | 489.7K | 125.6% |
| 15 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 99.5% | 0.052 | DEXEUSDT | 100.0% | 0.3 | 240.7M | 761.5% |
| 16 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 99.4% | 0.070 | STABLEUSDT | 100.0% | 0.8 | 10.2M | 241.6% |
| 17 | US | USUSDT (usdm-futures) | usdm-futures | 99.4% | 0.046 | ONEUSDT | 100.0% | 0 | 59.6M | 306.0% |
| 18 | AI | AIUSDT (spot) | spot | 99.3% | 0.079 | BTCUSDT | 100.0% | 3.5 | 291.5K | 95.5% |
| 19 | U | UUSDT (spot) | spot | 99.2% | 0.058 | WENUSDT | 100.0% | 3.8 | 14.9M | 1.3% |
| 20 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 99.2% | 0.088 | BTCUSDT | 100.0% | 0.3 | 24.8M | 240.5% |
| 21 | SYN | SYNUSDT (spot) | spot, usdm-futures | 99.0% | 0.047 | USUSDT | 100.0% | 0.3 | 5.6M | 252.7% |
| 22 | JST | JSTUSDT (spot) | spot, usdm-futures | 98.9% | 0.065 | DEXEUSDT | 100.0% | 0.5 | 2.2M | 34.3% |
| 23 | T | TUSDT (spot) | spot, usdm-futures | 98.8% | 0.070 | BTCUSDT | 100.0% | 1.5 | 1.6M | 145.9% |
| 24 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 98.8% | 0.077 | USUSDT | 100.0% | 1.3 | 400K | 61.3% |
| 25 | KITE | KITEUSDT (spot) | spot, usdm-futures | 98.8% | 0.109 | BTCUSDT | 100.0% | 0.8 | 28.4M | 109.7% |
| 26 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 98.7% | 0.085 | TURTLEUSDT | 100.0% | 0.3 | 2.2M | 87.7% |
| 27 | G | GUSDT (spot) | spot, usdm-futures | 98.6% | 0.101 | BTCUSDT | 100.0% | 2 | 599.9K | 238.7% |
| 28 | TTWO | TTWOUSDT (usdm-futures) | usdm-futures | 98.5% | 0.100 | BTCUSDT | 100.0% | 0.5 | 200.7K | 34.8% |
| 29 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 98.2% | 0.072 | DEXEUSDT | 100.0% | 0.3 | 2M | 148.5% |
| 30 | M | MUSDT (usdm-futures) | usdm-futures | 98.2% | 0.112 | BTCUSDT | 100.0% | 0.3 | 2.1M | 81.2% |
| 31 | TLM | TLMUSDT (spot) | spot, usdm-futures | 98.1% | 0.099 | EPICUSDT | 100.0% | 0.5 | 4M | 339.0% |
| 32 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 98.1% | 0.091 | PHAROSUSDT | 100.0% | 0.3 | 1.6M | 244.7% |
| 33 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 98.1% | 0.092 | LUMIAUSDT | 100.0% | 0.3 | 3.5M | 151.4% |
| 34 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 97.9% | 0.110 | BTCUSDT | 100.0% | 0.8 | 2.3M | 94.8% |
| 35 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 97.9% | 0.096 | QIUSDT | 100.0% | 0.8 | 3.8M | 133.2% |
| 36 | UB | UBUSDT (usdm-futures) | usdm-futures | 97.7% | 0.100 | USUSDT | 100.0% | 0.3 | 20.1M | 195.6% |
| 37 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 97.6% | 0.085 | HOMEUSDT | 100.0% | 0.5 | 2M | 203.9% |
| 38 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 97.6% | 0.076 | BULLAUSDT | 100.0% | 0.5 | 8.7M | 171.7% |
| 39 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 97.5% | 0.110 | BTCUSDT | 100.0% | 0.5 | 4.8M | 176.9% |
| 40 | KGST | KGSTUSDT (spot) | spot | 97.5% | 0.098 | BLUAIUSDT | 100.0% | 3.5 | 2.5M | 1.8% |
| 41 | BR | BRUSDT (usdm-futures) | usdm-futures | 97.4% | 0.103 | TUSDT | 100.0% | 0 | 1.5M | 156.4% |
| 42 | OGN | OGNUSDT (spot) | spot, usdm-futures | 97.3% | 0.081 | QIUSDT | 100.0% | 0.8 | 318.2K | 114.6% |
| 43 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 97.2% | 0.135 | BTCUSDT | 100.0% | 0.5 | 1.9M | 139.2% |
| 44 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 97.1% | 0.146 | BTCUSDT | 100.0% | 0.5 | 558.8K | 111.1% |
| 45 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 97.0% | 0.096 | BTCUSDT | 100.0% | 0.3 | 653.8K | 185.7% |
| 46 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 97.0% | 0.100 | ONEUSDT | 100.0% | 0.3 | 6.1M | 157.4% |
| 47 | H | HUSDT (usdm-futures) | usdm-futures | 96.8% | 0.098 | BTCUSDT | 100.0% | 0.3 | 5.2M | 107.2% |
| 48 | BAR | BARUSDT (spot) | spot | 96.8% | 0.111 | TUSDT | 100.0% | 2 | 881K | 161.7% |
| 49 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 96.6% | 0.104 | AIUSDT | 100.0% | 0.3 | 17.3M | 362.4% |
| 50 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 96.6% | 0.140 | BTCUSDT | 100.0% | 0.3 | 2.1M | 119.9% |
| 51 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 96.5% | 0.144 | BTCUSDT | 100.0% | 0.3 | 1M | 166.7% |
| 52 | PIVX | PIVXUSDT (spot) | spot | 96.5% | 0.107 | FOLKSUSDT | 100.0% | 1.8 | 232.8K | 186.3% |
| 53 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 96.2% | 0.120 | JSTUSDT | 100.0% | 0.5 | 8.2M | 210.7% |
| 54 | DGB | DGBUSDT (spot) | spot | 96.1% | 0.080 | GUSDT | 100.0% | 1 | 898K | 377.9% |
| 55 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 96.0% | 0.095 | GUSDT | 100.0% | 0.5 | 793.7K | 72.5% |
| 56 | QUICK | QUICKUSDT (spot) | spot | 96.0% | 0.118 | AMPUSDT | 100.0% | 0.8 | 43.3K | 90.8% |
| 57 | TREE | TREEUSDT (spot) | spot, usdm-futures | 95.8% | 0.152 | BTCUSDT | 100.0% | 1.3 | 2.3M | 192.2% |
| 58 | IBM | IBMBUSDT (spot) | spot, usdm-futures | 95.8% | 0.182 | BTCUSDT | 100.0% | 0.8 | 353.2K | 65.9% |
| 59 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 95.7% | 0.100 | BLUAIUSDT | 100.0% | 0.5 | 4.3M | 152.4% |
| 60 | YB | YBUSDT (spot) | spot, usdm-futures | 95.6% | 0.105 | BLUAIUSDT | 100.0% | 0.8 | 545.7K | 146.5% |
| 61 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 95.5% | 0.170 | ONEUSDT | 100.0% | 1 | 693.7K | 34.0% |
| 62 | O | OUSDT (usdm-futures) | usdm-futures | 95.4% | 0.153 | ENSOUSDT | 100.0% | 0.3 | 4.8M | 155.5% |
| 63 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 95.3% | 0.151 | MUSDT | 100.0% | 0.5 | 3.1M | 159.2% |
| 64 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 95.3% | 0.130 | BLUAIUSDT | 100.0% | 0.5 | 3.3M | 137.6% |
| 65 | B | BUSDT (usdm-futures) | usdm-futures | 95.2% | 0.107 | TURTLEUSDT | 100.0% | 0.3 | 24.3M | 327.5% |
| 66 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 95.1% | 0.144 | AMPUSDT | 100.0% | 0.8 | 785K | 118.9% |
| 67 | BAN | BANUSDT (usdm-futures) | usdm-futures | 95.0% | 0.123 | BTCUSDT | 100.0% | 0.8 | 1.3M | 74.7% |
| 68 | BABA | BABABUSDT (spot) | spot, usdm-futures | 94.9% | 0.113 | BTCUSDT | 100.0% | 0.8 | 216K | 60.9% |
| 69 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 94.9% | 0.119 | KITEUSDT | 100.0% | 0 | 341.9M | 526.6% |
| 70 | WET | WETUSDT (usdm-futures) | usdm-futures | 94.8% | 0.123 | BTCUSDT | 100.0% | 0.3 | 1.2M | 84.3% |
| 71 | LA | LAUSDT (spot) | spot, usdm-futures | 94.8% | 0.145 | BTCUSDT | 100.0% | 0.8 | 467.9K | 189.8% |
| 72 | ON | ONUSDT (usdm-futures) | usdm-futures | 94.4% | 0.110 | AGTUSDT | 100.0% | 0.3 | 13.2M | 336.0% |
| 73 | MMT | MMTUSDT (spot) | spot, usdm-futures | 94.3% | 0.123 | BRUSDT | 100.0% | 0.5 | 962K | 97.3% |
| 74 | AWE | AWEUSDT (spot) | spot, usdm-futures | 94.2% | 0.121 | SYNUSDT | 100.0% | 0.8 | 299K | 95.6% |
| 75 | REQ | REQUSDT (spot) | spot | 94.1% | 0.202 | BTCUSDT | 100.0% | 2 | 52.8K | 98.4% |
| 76 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 94.0% | 0.106 | FOLKSUSDT | 100.0% | 0.5 | 5.2M | 178.2% |
| 77 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 94.0% | 0.152 | HUSDT | 100.0% | 0.5 | 2.5M | 72.1% |
| 78 | TA | TAUSDT (usdm-futures) | usdm-futures | 93.9% | 0.122 | ALLOUSDT | 100.0% | 0.5 | 1.1M | 70.5% |
| 79 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 93.8% | 0.180 | BTCUSDT | 100.0% | 0.3 | 2.6M | 96.4% |
| 80 | ATM | ATMUSDT (spot) | spot | 93.6% | 0.160 | DEXEUSDT | 100.0% | 0.5 | 964.6K | 257.5% |
| 81 | MBL | MBLUSDT (spot) | spot | 93.6% | 0.170 | BTCUSDT | 100.0% | 1 | 398.2K | 61.5% |
| 82 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 93.4% | 0.145 | EVAAUSDT | 100.0% | 0.3 | 2.9M | 147.1% |
| 83 | LAZIO | LAZIOUSDT (spot) | spot | 93.4% | 0.128 | BARUSDT | 100.0% | 2.3 | 264.8K | 80.6% |
| 84 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 93.3% | 0.149 | BULLAUSDT | 100.0% | 0.3 | 5M | 261.8% |
| 85 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 93.3% | 0.133 | MUSDT | 100.0% | 1.3 | 187.9K | 30.5% |
| 86 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 93.1% | 0.097 | ONEUSDT | 100.0% | 0.8 | 305.1K | 36.8% |
| 87 | HEI | HEIUSDT (spot) | spot, usdm-futures | 92.9% | 0.147 | PTBUSDT | 100.0% | 0.5 | 1.6M | 165.6% |
| 88 | BEL | BELUSDT (spot) | spot, usdm-futures | 92.9% | 0.160 | BTCUSDT | 100.0% | 0.8 | 478.1K | 78.4% |
| 89 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 92.8% | 0.097 | TTWOUSDT | 100.0% | 1 | 18.3M | 39.2% |
| 90 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 92.7% | 0.115 | TREEUSDT | 100.0% | 0.3 | 2.6M | 144.5% |
| 91 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 92.5% | 0.118 | EVAAUSDT | 100.0% | 0.3 | 2M | 136.9% |
| 92 | PYR | PYRUSDT (spot) | spot | 92.4% | 0.149 | BROCCOLIF3BUSDT | 100.0% | 1.3 | 1.1M | 204.1% |
| 93 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 92.4% | 0.147 | BTCUSDT | 100.0% | 0.5 | 3.1M | 114.8% |
| 94 | STAR | STARUSDT (usdm-futures) | usdm-futures | 92.3% | 0.109 | TAGUSDT | 100.0% | 0.3 | 3.7M | 271.8% |
| 95 | PROM | PROMUSDT (spot) | spot, usdm-futures | 92.3% | 0.117 | TURTLEUSDT | 100.0% | 0.8 | 2.6M | 293.6% |
| 96 | SUN | SUNUSDT (spot) | spot, usdm-futures | 92.2% | 0.180 | BTCUSDT | 100.0% | 1.5 | 350.2K | 21.4% |
| 97 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 92.2% | 0.138 | BTCUSDT | 100.0% | 0.3 | 5.1M | 152.9% |
| 98 | TAC | TACUSDT (usdm-futures) | usdm-futures | 91.9% | 0.138 | ROBOUSDT | 100.0% | 0.3 | 8.4M | 239.9% |
| 99 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 91.8% | 0.174 | BTCUSDT | 100.0% | 1.3 | 205.5K | 35.3% |
| 100 | IQ | IQUSDT (spot) | spot | 91.5% | 0.147 | BTCUSDT | 100.0% | 1.8 | 26.4K | 49.3% |
| 101 | FLOCK | FLOCKUSDT (usdm-futures) | usdm-futures | 91.4% | 0.159 | BTCUSDT | 100.0% | 0.3 | 3.1M | 122.6% |
| 102 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 91.4% | 0.135 | BTCUSDT | 100.0% | 0.3 | 2.1M | 142.1% |
| 103 | RIF | RIFUSDT (spot) | spot, usdm-futures | 91.2% | 0.155 | DEXEUSDT | 100.0% | 0.5 | 1.4M | 317.8% |
| 104 | FTT | FTTUSDT (spot) | spot | 91.1% | 0.133 | BTCUSDT | 100.0% | 0.3 | 207.4K | 112.8% |
| 105 | GPS | GPSUSDT (spot) | spot, usdm-futures | 91.0% | 0.116 | BTCUSDT | 100.0% | 0.8 | 395.1K | 91.3% |
| 106 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 90.9% | 0.121 | CAPUSDT | 100.0% | 0.3 | 26.4M | 32.1% |
| 107 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 90.7% | 0.133 | BTCUSDT | 100.0% | 1 | 7.9M | 169.2% |
| 108 | DODO | DODOUSDT (spot) | spot | 90.6% | 0.108 | STABLEUSDT | 100.0% | 0.3 | 3.3M | 314.1% |
| 109 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 90.5% | 0.121 | YBUSDT | 100.0% | 0.8 | 342.8K | 33.4% |
| 110 | AIN | AINUSDT (usdm-futures) | usdm-futures | 90.4% | 0.132 | PIVXUSDT | 100.0% | 0.3 | 833.2K | 123.5% |
| 111 | APR | APRUSDT (usdm-futures) | usdm-futures | 90.4% | 0.125 | TUSDT | 100.0% | 0.5 | 2M | 76.4% |
| 112 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 90.2% | 0.154 | AGTUSDT | 100.0% | 0.3 | 12.7M | 173.7% |
| 113 | CC | CCUSDT (usdm-futures) | usdm-futures | 90.1% | 0.172 | BTCUSDT | 100.0% | 0.3 | 3.2M | 52.2% |
| 114 | TKO | TKOUSDT (spot) | spot | 90.0% | 0.195 | BTCUSDT | 100.0% | 1.3 | 79.6K | 79.8% |
| 115 | BAS | BASUSDT (usdm-futures) | usdm-futures | 90.0% | 0.160 | BTCUSDT | 100.0% | 0.3 | 2.2M | 116.0% |
| 116 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 89.8% | 0.129 | JCTUSDT | 100.0% | 0.5 | 969.2K | 100.8% |
| 117 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 89.8% | 0.180 | BTCUSDT | 100.0% | 0.8 | 1.3M | 83.0% |
| 118 | SENT | SENTUSDT (spot) | spot, usdm-futures | 89.8% | 0.137 | BTCUSDT | 100.0% | 0.8 | 1.4M | 95.3% |
| 119 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 89.5% | 0.163 | BTCUSDT | 100.0% | 0.3 | 1.8M | 126.2% |
| 120 | RPL | RPLUSDT (spot) | spot, usdm-futures | 89.5% | 0.211 | BTCUSDT | 100.0% | 1.3 | 258.9K | 108.9% |
| 121 | VANA | VANAUSDT (spot) | spot, usdm-futures | 89.4% | 0.167 | 龙虾USDT | 100.0% | 0.8 | 1.2M | 96.6% |
| 122 | TUT | TUTUSDT (spot) | spot, usdm-futures | 89.2% | 0.136 | BTCUSDT | 100.0% | 0.3 | 471K | 110.9% |
| 123 | RARE | RAREUSDT (spot) | spot, usdm-futures | 89.1% | 0.156 | BTCUSDT | 100.0% | 6 | 153.7K | 84.0% |
| 124 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 88.9% | 0.141 | BTCUSDT | 100.0% | 0.5 | 1.6M | 108.1% |
| 125 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 88.9% | 0.201 | BTCUSDT | 100.0% | 0.8 | 150.7K | 78.3% |
| 126 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 88.8% | 0.158 | TACUSDT | 100.0% | 0.5 | 9.5M | 162.6% |
| 127 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 88.7% | 0.171 | BTCUSDT | 100.0% | 0.3 | 3.8M | 216.7% |
| 128 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 88.5% | 0.120 | ICNTUSDT | 100.0% | 0.5 | 611.3K | 100.2% |
| 129 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 88.4% | 0.117 | JSTUSDT | 100.0% | 0.3 | 2.6M | 140.3% |
| 130 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 88.4% | 0.156 | BTCUSDT | 100.0% | 1.8 | 3.4M | 101.6% |
| 131 | KAVA | KAVAUSDT (spot) | spot, usdm-futures | 88.3% | 0.232 | BTCUSDT | 100.0% | 0.8 | 261.7K | 52.5% |
| 132 | LISTA | LISTAUSDT (spot) | spot, usdm-futures | 88.3% | 0.208 | BTCUSDT | 100.0% | 2.3 | 149.6K | 151.4% |
| 133 | ZIL | ZILUSDT (spot) | spot, usdm-futures | 88.2% | 0.227 | BTCUSDT | 100.0% | 2 | 651K | 97.3% |
| 134 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 88.2% | 0.172 | XPINUSDT | 100.0% | 0.3 | 8.8M | 171.6% |
| 135 | Q | QUSDT (usdm-futures) | usdm-futures | 87.9% | 0.151 | CYSUSDT | 100.0% | 0.3 | 1.1M | 85.3% |
| 136 | OPN | OPNUSDT (spot) | spot, usdm-futures | 87.9% | 0.133 | FRAXUSDT | 100.0% | 1.3 | 15.1M | 115.4% |
| 137 | JUV | JUVUSDT (spot) | spot | 87.7% | 0.177 | BARUSDT | 100.0% | 2 | 314.8K | 92.6% |
| 138 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 87.7% | 0.139 | SONYUSDT | 100.0% | 0.5 | 2.7M | 168.7% |
| 139 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 87.7% | 0.243 | BTCUSDT | 100.0% | 1 | 2.2M | 143.3% |
| 140 | GNS | GNSUSDT (spot) | spot | 87.4% | 0.137 | TURTLEUSDT | 100.0% | 2.5 | 55.2K | 49.3% |
| 141 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 87.4% | 0.144 | CYSUSDT | 100.0% | 0.8 | 3.6M | 102.7% |
| 142 | STRAX | STRAXUSDT (spot) | spot | 87.2% | 0.219 | BTCUSDT | 100.0% | 1.5 | 355.8K | 93.7% |
| 143 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 87.1% | 0.205 | TTWOUSDT | 100.0% | 0.8 | 295.9K | 55.7% |
| 144 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 87.0% | 0.123 | AIOTUSDT | 100.0% | 0.5 | 1.3M | 111.0% |
| 145 | FF | FFUSDT (spot) | spot, usdm-futures | 86.9% | 0.129 | BSBUSDT | 100.0% | 0.5 | 1.1M | 59.4% |
| 146 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 86.8% | 0.147 | GUSDT | 100.0% | 0.5 | 2.5M | 148.6% |
| 147 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 86.5% | 0.129 | ONEUSDT | 100.0% | 0.8 | 3M | 145.1% |
| 148 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 86.4% | 0.181 | BTCUSDT | 100.0% | 0.3 | 1M | 111.8% |
| 149 | TST | TSTUSDT (spot) | spot, usdm-futures | 86.2% | 0.177 | BTCUSDT | 100.0% | 1 | 309.9K | 89.0% |
| 150 | LAB | LABUSDT (usdm-futures) | usdm-futures | 86.1% | 0.178 | TURTLEUSDT | 100.0% | 0.5 | 97M | 312.1% |
| 151 | XAN | XANUSDT (usdm-futures) | usdm-futures | 85.9% | 0.146 | TAKEUSDT | 100.0% | 0.3 | 3.5M | 173.5% |
| 152 | SOLV | SOLVUSDT (spot) | spot, usdm-futures | 85.7% | 0.171 | BTCUSDT | 100.0% | 4 | 285.2K | 146.7% |
| 153 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 85.7% | 0.186 | BTCUSDT | 100.0% | 0.5 | 10.3M | 132.4% |
| 154 | IWM | IWMUSDT (usdm-futures) | usdm-futures | 85.5% | 0.225 | BTCUSDT | 100.0% | 1.3 | 237.3K | 30.0% |
| 155 | XEC | XECUSDT (spot) | spot, usdm-futures | 85.4% | 0.122 | AKEUSDT | 100.0% | 0.5 | 3.6M | 256.1% |
| 156 | WIN | WINUSDT (spot) | spot | 85.4% | 0.224 | BTCUSDT | 100.0% | 0.8 | 98.5K | 64.1% |
| 157 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 85.2% | 0.174 | BRKBUSDT | 100.0% | 0.3 | 1.5M | 124.4% |
| 158 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 85.1% | 0.170 | STRAXUSDT | 100.0% | 0.3 | 4.3M | 152.7% |
| 159 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 85.0% | 0.131 | SENTUSDT | 100.0% | 0.3 | 1.8M | 106.9% |
| 160 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 84.9% | 0.125 | LYNUSDT | 100.0% | 0.3 | 23.9M | 223.6% |
| 161 | ESP | ESPUSDT (spot) | spot, usdm-futures | 84.9% | 0.150 | ENSOUSDT | 100.0% | 0.3 | 256.9K | 56.4% |
| 162 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 84.6% | 0.173 | BTCUSDT | 100.0% | 0.5 | 9.1M | 134.5% |
| 163 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 84.5% | 0.268 | BTCUSDT | 100.0% | 0.3 | 1.2M | 93.7% |
| 164 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 84.4% | 0.219 | BTCUSDT | 100.0% | 5 | 165.8K | 37.9% |
| 165 | QKC | QKCUSDT (spot) | spot | 84.3% | 0.150 | AAPLUSDT | 100.0% | 0.8 | 70.5K | 68.2% |
| 166 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 84.3% | 0.128 | AERGOUSDT | 100.0% | 1.5 | 6.4M | 183.2% |
| 167 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 84.0% | 0.172 | BTCUSDT | 100.0% | 0.3 | 630.8K | 73.4% |
| 168 | A | AUSDT (spot) | spot, usdm-futures | 84.0% | 0.235 | BTCUSDT | 100.0% | 0.8 | 295.6K | 90.6% |
| 169 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 83.9% | 0.256 | BTCUSDT | 100.0% | 0.5 | 574K | 87.5% |
| 170 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 83.9% | 0.262 | BTCUSDT | 100.0% | 1 | 766.3K | 78.4% |
| 171 | MET | METUSDT (spot) | spot, usdm-futures | 83.7% | 0.194 | BTCUSDT | 100.0% | 0.5 | 915.3K | 129.1% |
| 172 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 83.6% | 0.130 | FRAXUSDT | 100.0% | 2.5 | 105.1K | 151.9% |
| 173 | BTTC | BTTCUSDT (spot) | spot | 83.4% | 0.113 | CROSSUSDT | 100.0% | 8.5 | 139.3K | 396.5% |
| 174 | B2 | B2USDT (usdm-futures) | usdm-futures | 83.3% | 0.219 | PHAROSUSDT | 100.0% | 0.3 | 710.7K | 415.8% |
| 175 | SC | SCUSDT (spot) | spot | 83.1% | 0.303 | BTCUSDT | 100.0% | 1.5 | 117.2K | 53.2% |
| 176 | RAD | RADUSDT (spot) | spot | 83.0% | 0.199 | AMPUSDT | 100.0% | 5 | 139.4K | 67.7% |
| 177 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 83.0% | 0.231 | BTCUSDT | 100.0% | 0.8 | 1.3M | 53.8% |
| 178 | ACX | ACXUSDT (spot) | spot, usdm-futures | 82.9% | 0.264 | BTCUSDT | 100.0% | 1 | 70.6K | 36.6% |
| 179 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 82.8% | 0.179 | ICNTUSDT | 100.0% | 1.5 | 1.1M | 119.9% |
| 180 | SKL | SKLUSDT (spot) | spot, usdm-futures | 82.6% | 0.181 | JCTUSDT | 100.0% | 1.3 | 930.6K | 140.3% |
| 181 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 82.5% | 0.186 | LAUSDT | 100.0% | 0.3 | 2.5M | 103.1% |
| 182 | OPENAI | OPENAIUSDT (usdm-futures) | usdm-futures | 82.3% | 0.341 | ANTHROPICUSDT | 100.0% | 0.3 | 3.3M | 77.6% |
| 183 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 82.3% | 0.260 | GUSDT | 100.0% | 0.3 | 2.3M | 167.1% |
| 184 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 82.1% | 0.162 | BTCUSDT | 100.0% | 0.3 | 10.4M | 134.2% |
| 185 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 82.0% | 0.214 | BTCUSDT | 100.0% | 0.5 | 795.3K | 91.6% |
| 186 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 81.9% | 0.196 | SOLVUSDT | 100.0% | 0.3 | 1.1M | 87.6% |
| 187 | SMCI | SMCIUSDT (usdm-futures) | usdm-futures | 81.7% | 0.258 | BRKBUSDT | 100.0% | 1 | 898.3K | 170.2% |
| 188 | SXT | SXTUSDT (spot) | spot, usdm-futures | 81.7% | 0.156 | BTCUSDT | 100.0% | 0.8 | 902.3K | 109.2% |
| 189 | THE | THEUSDT (spot) | spot, usdm-futures | 81.7% | 0.212 | BTCUSDT | 100.0% | 1.3 | 521K | 100.2% |
| 190 | ACE | ACEUSDT (spot) | spot, usdm-futures | 81.6% | 0.159 | AIAUSDT | 100.0% | 1 | 4.4M | 349.5% |
| 191 | MTL | MTLUSDT (spot) | spot, usdm-futures | 81.6% | 0.258 | BTCUSDT | 100.0% | 3.5 | 25.7K | 54.6% |
| 192 | DCR | DCRUSDT (spot) | spot | 81.3% | 0.217 | BTCUSDT | 100.0% | 1 | 250.6K | 77.0% |
| 193 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 81.2% | 0.204 | NVOUSDT | 100.0% | 0.3 | 888.8K | 66.8% |
| 194 | GLMR | GLMRUSDT (spot) | spot | 81.1% | 0.104 | IWMUSDT | 100.0% | 2 | 288K | 148.2% |
| 195 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 80.9% | 0.267 | BTCUSDT | 100.0% | 0.8 | 522.4K | 92.7% |
| 196 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 80.6% | 0.162 | SPORTFUNUSDT | 100.0% | 1 | 347.5K | 82.3% |
| 197 | HOT | HOTUSDT (spot) | spot, usdm-futures | 80.5% | 0.244 | BTCUSDT | 100.0% | 1.5 | 141.5K | 88.3% |
| 198 | HFT | HFTUSDT (spot) | spot, usdm-futures | 80.3% | 0.192 | BTCUSDT | 100.0% | 5.8 | 92.8K | 138.6% |
| 199 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 80.3% | 0.178 | BTCUSDT | 100.0% | 0.3 | 556.3K | 53.6% |
| 200 | AUDIO | AUDIOUSDT (spot) | spot | 80.1% | 0.169 | BTCUSDT | 100.0% | 0.8 | 366.2K | 77.9% |
| 201 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 80.0% | 0.188 | BTCUSDT | 100.0% | 0.5 | 2.2M | 69.0% |
| 202 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 79.9% | 0.233 | BTCUSDT | 100.0% | 0.8 | 682.2K | 96.4% |
| 203 | XVS | XVSUSDT (spot) | spot, usdm-futures | 79.8% | 0.320 | BTCUSDT | 100.0% | 2.5 | 90.6K | 64.9% |
| 204 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 79.7% | 0.263 | BTCUSDT | 100.0% | 0.3 | 1.7M | 65.0% |
| 205 | RE | REUSDT (spot) | spot, usdm-futures | 79.6% | 0.149 | BASUSDT | 100.0% | 0.3 | 43.3M | 154.7% |
| 206 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 79.6% | 0.272 | BROCCOLIF3BUSDT | 100.0% | 0.5 | 5.3M | 274.8% |
| 207 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 79.5% | 0.147 | TLMUSDT | 100.0% | 0.3 | 5.8M | 184.4% |
| 208 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 79.4% | 0.207 | BTCUSDT | 100.0% | 0.5 | 1.3M | 79.9% |
| 209 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 79.3% | 0.275 | BTCUSDT | 100.0% | 0.5 | 592.7K | 75.1% |
| 210 | GNO | GNOUSDT (spot) | spot | 79.2% | 0.307 | BTCUSDT | 100.0% | 0.3 | 167.6K | 56.5% |
| 211 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 79.1% | 0.231 | TACUSDT | 100.0% | 0.3 | 5.4M | 147.9% |
| 212 | GTC | GTCUSDT (spot) | spot, usdm-futures | 79.0% | 0.135 | DATAIPUSDT | 100.0% | 3.3 | 119.8K | 168.7% |
| 213 | CATI | CATIUSDT (spot) | spot, usdm-futures | 78.9% | 0.202 | BTCUSDT | 100.0% | 0.3 | 266.9K | 90.1% |
| 214 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 78.6% | 0.160 | LABUSDT | 100.0% | 0.3 | 14M | 141.2% |
| 215 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 78.3% | 0.236 | BTCUSDT | 100.0% | 0.5 | 1.3M | 97.4% |
| 216 | ERA | ERAUSDT (spot) | spot, usdm-futures | 78.1% | 0.266 | LAUSDT | 100.0% | 1.3 | 856.9K | 380.1% |
| 217 | ONG | ONGUSDT (spot) | spot, usdm-futures | 78.1% | 0.302 | BTCUSDT | 100.0% | 0.8 | 182.5K | 63.5% |
| 218 | LDO | LDOUSDT (spot) | spot, usdm-futures | 77.8% | 0.334 | BTCUSDT | 100.0% | 0.3 | 3.7M | 90.7% |
| 219 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 77.7% | 0.202 | LABUSDT | 100.0% | 0.3 | 18.8M | 138.6% |
| 220 | USELESS | USELESSUSDT (usdm-futures) | usdm-futures | 77.6% | 0.319 | BTCUSDT | 100.0% | 0.3 | 12M | 151.6% |
| 221 | 0G | 0GUSDT (spot) | spot, usdm-futures | 77.5% | 0.208 | BTCUSDT | 100.0% | 2.5 | 1.5M | 109.0% |
| 222 | CTSI | CTSIUSDT (spot) | spot, usdm-futures | 77.4% | 0.303 | BTCUSDT | 100.0% | 1 | 95.6K | 66.5% |
| 223 | KERNEL | KERNELUSDT (spot) | spot, usdm-futures | 77.4% | 0.244 | TREEUSDT | 100.0% | 2 | 251.4K | 180.8% |
| 224 | AT | ATUSDT (spot) | spot, usdm-futures | 77.1% | 0.187 | BTCUSDT | 100.0% | 0.8 | 213.7K | 69.8% |
| 225 | MINA | MINAUSDT (spot) | spot, usdm-futures | 76.9% | 0.287 | BTCUSDT | 100.0% | 1.8 | 333.5K | 75.4% |
| 226 | XLE | XLEUSDT (usdm-futures) | usdm-futures | 76.8% | 0.113 | IDOLUSDT | 100.0% | 0.5 | 158.5K | 25.8% |
| 227 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 76.7% | 0.288 | BTCUSDT | 100.0% | 0.8 | 160.4K | 50.9% |
| 228 | PYTH | PYTHUSDT (spot) | spot, usdm-futures | 76.7% | 0.291 | BTCUSDT | 100.0% | 0.3 | 1.8M | 77.2% |
| 229 | OSMO | OSMOUSDT (spot) | spot | 76.5% | 0.301 | BTCUSDT | 100.0% | 1.8 | 153.8K | 68.9% |
| 230 | NOM | NOMUSDT (spot) | spot, usdm-futures | 76.2% | 0.201 | BTCUSDT | 100.0% | 2.8 | 454.7K | 101.3% |
| 231 | INIT | INITUSDT (spot) | spot, usdm-futures | 76.2% | 0.293 | BTCUSDT | 100.0% | 2 | 63.7K | 68.7% |
| 232 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 76.0% | 0.239 | BTCUSDT | 100.0% | 0.5 | 1.2M | 108.9% |
| 233 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 75.7% | 0.196 | REQUSDT | 100.0% | 0.3 | 4.1M | 143.5% |
| 234 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 75.7% | 0.180 | BTCUSDT | 100.0% | 0.5 | 841.4K | 90.1% |
| 235 | V | VUSDT (usdm-futures) | usdm-futures | 75.3% | 0.197 | ONEUSDT | 100.0% | 0.5 | 296.9K | 22.9% |
| 236 | MOVE | MOVEUSDT (spot) | spot, usdm-futures | 75.3% | 0.262 | BTCUSDT | 100.0% | 12.3 | 351.9K | 70.6% |
| 237 | PHA | PHAUSDT (spot) | spot, usdm-futures | 75.1% | 0.303 | BTCUSDT | 100.0% | 1.8 | 743.5K | 119.7% |
| 238 | BX | BXUSDT (usdm-futures) | usdm-futures | 75.0% | 0.210 | CRWDUSDT | 100.0% | 0.5 | 804.9K | 47.7% |
| 239 | PLUME | PLUMEUSDT (spot) | spot, usdm-futures | 75.0% | 0.300 | BTCUSDT | 100.0% | 0.8 | 1.1M | 109.4% |
| 240 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 74.8% | 0.331 | BTCUSDT | 100.0% | 0.5 | 1.5M | 87.5% |
| 241 | NOK | NOKBUSDT (spot) | spot, usdm-futures | 74.7% | 0.331 | BTCUSDT | 100.0% | 1 | 154.8K | 80.2% |
| 242 | BZ | BZUSDT (usdm-futures) | usdm-futures | 74.5% | 0.235 | XLEUSDT | 100.0% | 0.5 | 259.2M | 43.6% |
| 243 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 74.4% | 0.242 | BTCUSDT | 100.0% | 0.3 | 418.9K | 78.3% |
| 244 | POL | POLUSDT (spot) | spot, usdm-futures | 74.3% | 0.253 | BTCUSDT | 100.0% | 0.5 | 2.3M | 40.0% |
| 245 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 74.1% | 0.191 | BTCUSDT | 100.0% | 0.3 | 1.5M | 66.2% |
| 246 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 74.0% | 0.301 | BTCUSDT | 100.0% | 0.5 | 347.2K | 87.8% |
| 247 | GRAM | GRAMUSDT (spot) | spot, usdm-futures | 73.8% | 0.338 | BTCUSDT | 100.0% | 0.5 | 6.1M | 90.7% |
| 248 | SPK | SPKUSDT (spot) | spot, usdm-futures | 73.8% | 0.253 | BTCUSDT | 100.0% | 0.3 | 913.1K | 57.0% |
| 249 | GME | GMEUSDT (usdm-futures) | usdm-futures | 73.7% | 0.247 | BTCUSDT | 100.0% | 2 | 166K | 26.5% |
| 250 | BONK | BONKUSDT (spot) | spot, usdm-futures | 73.6% | 0.331 | BTCUSDT | 100.0% | 1 | 6.2M | 132.0% |
| 251 | POLYX | POLYXUSDT (spot) | spot, usdm-futures | 73.5% | 0.275 | BTCUSDT | 100.0% | 2 | 87.9K | 73.0% |
| 252 | KNC | KNCUSDT (spot) | spot, usdm-futures | 73.4% | 0.282 | KAVAUSDT | 100.0% | 1.5 | 112.8K | 99.3% |
| 253 | BTR | BTRUSDT (usdm-futures) | usdm-futures | 73.2% | 0.319 | BTCUSDT | 100.0% | 0.8 | 1M | 86.6% |
| 254 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 73.2% | 0.239 | SPORTFUNUSDT | 100.0% | 0.8 | 1.4M | 79.5% |
| 255 | TRX | TRXUSDT (spot) | spot, usdm-futures, coinm-futures | 73.0% | 0.270 | BTCUSDT | 100.0% | 1 | 21.4M | 12.4% |
| 256 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 72.9% | 0.286 | BTCUSDT | 100.0% | 1 | 863.5K | 101.8% |
| 257 | ID | IDUSDT (spot) | spot, usdm-futures | 72.8% | 0.223 | BTCUSDT | 100.0% | 1.5 | 425.6K | 79.9% |
| 258 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 72.7% | 0.351 | BTCUSDT | 100.0% | 1 | 183.8K | 49.3% |
| 259 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 72.6% | 0.205 | DATAIPUSDT | 100.0% | 0.8 | 796K | 118.8% |
| 260 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 72.5% | 0.222 | BTCUSDT | 100.0% | 1.8 | 3.2M | 45.6% |
| 261 | SNX | SNXUSDT (spot) | spot, usdm-futures | 72.3% | 0.263 | BTCUSDT | 100.0% | 2.3 | 960.1K | 100.0% |
| 262 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 72.1% | 0.217 | SPKUSDT | 100.0% | 0.3 | 2M | 129.2% |
| 263 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 72.0% | 0.209 | PYTHUSDT | 100.0% | 0.5 | 371.4K | 93.4% |
| 264 | MELANIA | MELANIAUSDT (usdm-futures) | usdm-futures | 71.9% | 0.322 | BTCUSDT | 100.0% | 0.3 | 750.8K | 67.5% |
| 265 | PUMP | PUMPUSDT (spot) | spot, usdm-futures | 71.9% | 0.361 | BTCUSDT | 100.0% | 0.5 | 11.2M | 118.4% |
| 266 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 71.5% | 0.239 | BTCUSDT | 100.0% | 0.5 | 2M | 83.6% |
| 267 | NMR | NMRUSDT (spot) | spot, usdm-futures | 71.4% | 0.342 | BTCUSDT | 100.0% | 1.3 | 184.5K | 47.6% |
| 268 | XAI | XAIUSDT (spot) | spot, usdm-futures | 71.2% | 0.339 | BTCUSDT | 100.0% | 1 | 283.1K | 80.2% |
| 269 | ASTER | ASTERUSDT (spot) | spot, usdm-futures | 71.2% | 0.409 | BTCUSDT | 100.0% | 1.5 | 2.5M | 34.1% |
| 270 | VTHO | VTHOUSDT (spot) | spot, usdm-futures | 71.1% | 0.401 | BTCUSDT | 100.0% | 2.5 | 121.7K | 51.1% |
| 271 | C | CUSDT (spot) | spot, usdm-futures | 70.9% | 0.353 | BTCUSDT | 100.0% | 1.5 | 227.5K | 76.5% |
| 272 | META | METABUSDT (spot) | spot, usdm-futures | 70.9% | 0.289 | TTWOUSDT | 100.0% | 0.5 | 121.4K | 43.1% |
| 273 | ACT | ACTUSDT (spot) | spot, usdm-futures | 70.6% | 0.219 | BTCUSDT | 100.0% | 0.5 | 529.3K | 100.3% |
| 274 | MITO | MITOUSDT (spot) | spot, usdm-futures | 70.5% | 0.228 | PLAYUSDT | 100.0% | 0.3 | 933.2K | 116.6% |
| 275 | CAT | 1000CATUSDT (spot) | spot, usdm-futures | 70.5% | 0.279 | BTCUSDT | 100.0% | 3.3 | 80.9K | 101.5% |
| 276 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 70.5% | 0.289 | BTCUSDT | 100.0% | 2 | 373.1K | 64.9% |
| 277 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 70.3% | 0.185 | PYTHUSDT | 100.0% | 0.5 | 9.1M | 164.6% |
| 278 | 我踏马来了 | 我踏马来了USDT (usdm-futures) | usdm-futures | 70.1% | 0.346 | BTCUSDT | 100.0% | 0.5 | 622.1K | 69.9% |
| 279 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 70.0% | 0.318 | STRAXUSDT | 100.0% | 1.5 | 3.3M | 183.9% |
| 280 | ACM | ACMUSDT (spot) | spot | 70.0% | 0.328 | JUVUSDT | 100.0% | 1.8 | 318.7K | 83.2% |
| 281 | GMX | GMXUSDT (spot) | spot, usdm-futures | 69.6% | 0.313 | BTCUSDT | 100.0% | 1.3 | 232K | 66.9% |
| 282 | XBI | XBIUSDT (usdm-futures) | usdm-futures | 69.5% | 0.211 | BTCUSDT | 100.0% | 0.5 | 391.5K | 40.9% |
| 283 | SCRT | SCRTUSDT (spot) | spot, usdm-futures | 69.4% | 0.289 | PHAUSDT | 100.0% | 1 | 171.1K | 84.0% |
| 284 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 69.3% | 0.299 | BTCUSDT | 100.0% | 0.5 | 681.2K | 60.0% |
| 285 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 69.2% | 0.295 | BTCUSDT | 100.0% | 0.3 | 2.3M | 95.5% |
| 286 | GEV | GEVUSDT (usdm-futures) | usdm-futures | 69.2% | 0.276 | NOKBUSDT | 100.0% | 0.8 | 374.5K | 92.5% |
| 287 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 69.1% | 0.224 | LISTAUSDT | 100.0% | 1.3 | 518K | 102.6% |
| 288 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 68.9% | 0.251 | BTCUSDT | 100.0% | 0.3 | 502.8K | 96.4% |
| 289 | STRC | STRCUSDT (usdm-futures) | usdm-futures | 68.6% | 0.301 | BTCUSDT | 100.0% | 0.8 | 1.1M | 29.8% |
| 290 | NEXO | NEXOUSDT (spot) | spot | 68.4% | 0.318 | BTCUSDT | 100.0% | 1.8 | 430K | 52.5% |
| 291 | ELSA | ELSAUSDT (usdm-futures) | usdm-futures | 68.3% | 0.231 | BTCUSDT | 100.0% | 0.3 | 3.8M | 87.6% |
| 292 | ADX | ADXUSDT (spot) | spot | 68.3% | 0.333 | BTCUSDT | 100.0% | 2.3 | 222.6K | 54.0% |
| 293 | KAIA | KAIAUSDT (spot) | spot, usdm-futures | 68.2% | 0.313 | BTCUSDT | 100.0% | 4 | 305.3K | 56.2% |
| 294 | SKY | SKYUSDT (spot) | spot, usdm-futures | 68.1% | 0.371 | BTCUSDT | 100.0% | 0.3 | 799.8K | 66.8% |
| 295 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 68.0% | 0.227 | COAIUSDT | 100.0% | 0.3 | 5.3M | 169.4% |
| 296 | OPG | OPGUSDT (spot) | spot, usdm-futures | 67.9% | 0.295 | BTCUSDT | 100.0% | 0.5 | 785.7K | 87.3% |
| 297 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 67.7% | 0.314 | BTCUSDT | 100.0% | 0.5 | 620.7K | 83.9% |
| 298 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 67.7% | 0.324 | BTCUSDT | 100.0% | 0.5 | 975.2K | 68.2% |
| 299 | IOTX | IOTXUSDT (spot) | spot, usdm-futures | 67.4% | 0.309 | BTCUSDT | 100.0% | 2.8 | 84.8K | 69.8% |
| 300 | ZRO | ZROUSDT (spot) | spot, usdm-futures | 67.3% | 0.340 | BTCUSDT | 100.0% | 0.8 | 1.9M | 79.6% |
| 301 | CITY | CITYUSDT (spot) | spot | 67.1% | 0.276 | JUVUSDT | 100.0% | 2.5 | 308.4K | 64.3% |
| 302 | AVGO | AVGOBUSDT (spot) | spot, usdm-futures | 66.8% | 0.315 | BTCUSDT | 100.0% | 0.5 | 92.1K | 57.0% |
| 303 | BIO | BIOUSDT (spot) | spot, usdm-futures | 66.8% | 0.413 | BTCUSDT | 100.0% | 0.5 | 993K | 76.4% |
| 304 | PORTO | PORTOUSDT (spot) | spot | 66.7% | 0.475 | LAZIOUSDT | 100.0% | 0.8 | 741K | 261.3% |
| 305 | ARK | ARKUSDT (spot) | spot, usdm-futures | 66.4% | 0.403 | BTCUSDT | 100.0% | 1.3 | 45.6K | 51.9% |
| 306 | KAT | KATUSDT (spot) | spot, usdm-futures | 66.4% | 0.275 | KMNOUSDT | 100.0% | 1.3 | 436.6K | 91.2% |
| 307 | AZTEC | AZTECUSDT (usdm-futures) | usdm-futures | 66.0% | 0.434 | BTCUSDT | 100.0% | 0.8 | 545.8K | 90.0% |
| 308 | ICX | ICXUSDT (spot) | spot, usdm-futures | 65.9% | 0.367 | BTCUSDT | 100.0% | 5 | 41.2K | 58.8% |
| 309 | QNT | QNTUSDT (spot) | spot, usdm-futures | 65.8% | 0.440 | BTCUSDT | 100.0% | 0.5 | 576K | 43.3% |
| 310 | XPD | XPDUSDT (usdm-futures) | usdm-futures | 65.6% | 0.268 | BTCUSDT | 100.0% | 0.3 | 3.7M | 41.1% |
| 311 | TFUEL | TFUELUSDT (spot) | spot | 65.5% | 0.221 | BTCUSDT | 100.0% | 1.5 | 104.3K | 63.8% |
| 312 | BANK | BANKUSDT (spot) | spot, usdm-futures | 65.4% | 0.166 | MYXUSDT | 100.0% | 0.3 | 77.1M | 767.5% |
| 313 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 65.3% | 0.298 | BTCUSDT | 100.0% | 0.8 | 1.1M | 57.5% |
| 314 | SPY | SPYBUSDT (spot) | spot, usdm-futures | 65.3% | 0.303 | BTCUSDT | 100.0% | 0.8 | 65.4K | 16.5% |
| 315 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 65.1% | 0.281 | BTCUSDT | 100.0% | 0.5 | 9.3M | 94.4% |
| 316 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 64.9% | 0.302 | AVGOBUSDT | 100.0% | 0.8 | 339.3K | 24.6% |
| 317 | TXN | TXNUSDT (usdm-futures) | usdm-futures | 64.7% | 0.276 | STRCUSDT | 100.0% | 0.5 | 402.9K | 57.6% |
| 318 | HAEDAL | HAEDALUSDT (spot) | spot, usdm-futures | 64.5% | 0.251 | BTCUSDT | 100.0% | 0.8 | 105.4K | 110.1% |
| 319 | BANANA | BANANAUSDT (spot) | spot, usdm-futures | 64.4% | 0.253 | BTCUSDT | 100.0% | 0.3 | 289.2K | 79.1% |
| 320 | CFG | CFGUSDT (spot) | spot, usdm-futures | 64.4% | 0.271 | BTCUSDT | 100.0% | 0.5 | 1.3M | 100.0% |
| 321 | ENS | ENSUSDT (spot) | spot, usdm-futures | 64.1% | 0.385 | BTCUSDT | 100.0% | 1 | 1M | 81.4% |

## Diagnostics

- Basis size selected: 321
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.055
- Maximum pairwise absolute correlation: 0.475
- Mean whole-market projection R²: 83.6%
- Median whole-market projection R²: 80.4%
- 10th-percentile whole-market projection R²: 62.5%
- Minimum whole-market projection R²: 59.1%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13.7% | 9.2% | 0.5% | 0.0% |
| 5 | 14.8% | 10.0% | 1.1% | 0.1% |
| 10 | 16.6% | 11.4% | 2.0% | 0.6% |
| 15 | 18.1% | 12.4% | 2.8% | 1.1% |
| 20 | 19.6% | 13.7% | 3.7% | 1.9% |
| 25 | 21.2% | 14.8% | 4.8% | 2.6% |
| 30 | 22.8% | 16.1% | 5.8% | 3.7% |
| 35 | 24.2% | 17.2% | 6.8% | 4.6% |
| 40 | 25.6% | 18.4% | 7.7% | 5.2% |
| 45 | 27.3% | 19.8% | 8.7% | 6.0% |
| 50 | 28.9% | 20.9% | 9.7% | 6.9% |
| 55 | 30.2% | 22.0% | 10.6% | 7.9% |
| 60 | 31.8% | 23.5% | 11.6% | 8.8% |
| 65 | 33.2% | 24.9% | 12.6% | 9.5% |
| 70 | 34.5% | 26.4% | 13.4% | 10.2% |
| 75 | 35.9% | 27.9% | 14.3% | 11.6% |
| 80 | 37.1% | 28.6% | 15.3% | 12.4% |
| 85 | 38.4% | 29.7% | 16.2% | 13.4% |
| 90 | 39.8% | 31.0% | 17.5% | 14.5% |
| 95 | 41.0% | 31.7% | 18.6% | 14.9% |
| 100 | 42.2% | 33.0% | 19.5% | 16.4% |
| 105 | 43.4% | 34.3% | 20.5% | 17.3% |
| 110 | 45.0% | 35.8% | 21.5% | 18.3% |
| 115 | 46.1% | 36.8% | 22.7% | 19.3% |
| 120 | 47.4% | 37.9% | 23.9% | 20.1% |
| 125 | 48.6% | 39.3% | 24.8% | 21.2% |
| 130 | 49.7% | 40.3% | 25.6% | 22.0% |
| 135 | 50.9% | 41.4% | 26.6% | 22.8% |
| 140 | 52.1% | 42.5% | 27.6% | 23.7% |
| 145 | 53.2% | 43.4% | 28.7% | 24.7% |
| 150 | 54.3% | 44.4% | 29.6% | 26.3% |
| 155 | 55.5% | 45.3% | 30.8% | 27.1% |
| 160 | 56.4% | 46.6% | 31.7% | 27.9% |
| 165 | 57.5% | 47.5% | 32.9% | 29.0% |
| 170 | 58.5% | 48.6% | 33.8% | 30.0% |
| 175 | 59.4% | 49.5% | 34.8% | 31.0% |
| 180 | 60.3% | 50.0% | 35.6% | 31.9% |
| 185 | 61.2% | 50.9% | 36.4% | 33.0% |
| 190 | 62.5% | 52.6% | 37.6% | 33.4% |
| 195 | 63.6% | 53.7% | 38.8% | 35.0% |
| 200 | 64.5% | 54.6% | 39.7% | 36.0% |
| 205 | 65.5% | 55.5% | 40.6% | 36.7% |
| 210 | 66.4% | 56.6% | 41.6% | 37.5% |
| 215 | 67.3% | 57.6% | 42.7% | 39.0% |
| 220 | 68.4% | 59.0% | 43.6% | 40.0% |
| 225 | 69.3% | 59.8% | 44.5% | 41.1% |
| 230 | 70.2% | 60.7% | 45.3% | 42.0% |
| 235 | 71.1% | 61.7% | 46.3% | 43.3% |
| 240 | 71.9% | 62.7% | 47.4% | 44.2% |
| 245 | 72.8% | 64.2% | 48.5% | 45.3% |
| 250 | 73.6% | 65.2% | 49.3% | 45.9% |
| 255 | 74.3% | 65.7% | 50.2% | 46.9% |
| 260 | 75.1% | 66.6% | 51.0% | 47.7% |
| 265 | 75.8% | 67.6% | 51.9% | 48.9% |
| 270 | 76.6% | 68.4% | 52.8% | 49.7% |
| 275 | 77.3% | 69.4% | 53.8% | 50.3% |
| 280 | 78.0% | 70.4% | 55.2% | 51.5% |
| 285 | 78.7% | 71.6% | 56.2% | 52.2% |
| 290 | 79.4% | 72.5% | 57.1% | 53.3% |
| 295 | 80.1% | 73.3% | 57.7% | 53.9% |
| 300 | 80.7% | 74.4% | 58.9% | 55.0% |
| 305 | 81.4% | 76.2% | 59.5% | 55.9% |
| 310 | 82.2% | 77.0% | 60.6% | 57.1% |
| 315 | 82.8% | 78.3% | 61.6% | 57.9% |
| 320 | 83.5% | 79.2% | 62.4% | 59.0% |
| 321 | 83.6% | 80.4% | 62.5% | 59.1% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | DEXE | BULLA | WEN | EPIC | STABLE | ONE | QI | BLUAI | XNO | PHAROS | TURTLE | LUMIA | AMP | ESPORTS | HOME | US | AI | U | EVAA | SYN | JST | T | ENSO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.000 | -0.010 | 0.006 | 0.005 | 0.030 | -0.001 | -0.004 | 0.060 | -0.042 | 0.065 | 0.039 | 0.013 | 0.061 | 0.046 | 0.015 | 0.039 | 0.079 | 0.047 | 0.088 | -0.046 | 0.020 | 0.070 | 0.013 |
| DEXE | 0.000 | 1.000 | 0.002 | 0.010 | -0.016 | 0.012 | 0.034 | 0.025 | -0.012 | 0.018 | 0.021 | -0.013 | 0.042 | -0.023 | -0.052 | -0.007 | -0.013 | -0.022 | -0.021 | -0.020 | 0.039 | 0.065 | 0.029 | 0.013 |
| BULLA | -0.010 | 0.002 | 1.000 | -0.010 | -0.001 | 0.002 | 0.009 | -0.001 | 0.015 | -0.005 | -0.025 | -0.029 | 0.033 | 0.012 | 0.039 | -0.034 | 0.042 | 0.000 | -0.019 | -0.002 | -0.044 | -0.032 | 0.022 | 0.015 |
| WEN | 0.006 | 0.010 | -0.010 | 1.000 | -0.003 | -0.016 | -0.017 | 0.022 | -0.026 | 0.010 | -0.008 | 0.035 | -0.018 | -0.001 | 0.001 | -0.003 | 0.020 | 0.029 | -0.058 | -0.025 | -0.008 | 0.011 | 0.037 | -0.047 |
| EPIC | 0.005 | -0.016 | -0.001 | -0.003 | 1.000 | -0.001 | -0.012 | -0.049 | -0.005 | -0.042 | -0.022 | 0.016 | -0.000 | 0.011 | -0.016 | 0.028 | 0.045 | -0.022 | 0.038 | 0.048 | 0.011 | -0.047 | 0.048 | 0.039 |
| STABLE | 0.030 | 0.012 | 0.002 | -0.016 | -0.001 | 1.000 | 0.011 | 0.013 | -0.004 | -0.026 | -0.006 | 0.009 | -0.060 | -0.019 | -0.027 | 0.070 | -0.005 | -0.040 | -0.013 | -0.026 | -0.040 | 0.024 | 0.041 | 0.044 |
| ONE | -0.001 | 0.034 | 0.009 | -0.017 | -0.012 | 0.011 | 1.000 | 0.004 | -0.016 | -0.017 | -0.006 | -0.026 | -0.027 | -0.004 | -0.034 | 0.036 | -0.046 | -0.000 | -0.011 | 0.023 | -0.019 | -0.028 | -0.005 | 0.011 |
| QI | -0.004 | 0.025 | -0.001 | 0.022 | -0.049 | 0.013 | 0.004 | 1.000 | 0.016 | 0.027 | 0.018 | 0.041 | 0.004 | 0.054 | 0.018 | 0.011 | 0.005 | -0.005 | 0.037 | -0.010 | -0.032 | -0.002 | -0.033 | 0.048 |
| BLUAI | 0.060 | -0.012 | 0.015 | -0.026 | -0.005 | -0.004 | -0.016 | 0.016 | 1.000 | 0.012 | -0.006 | -0.015 | -0.022 | -0.008 | 0.029 | -0.037 | 0.026 | -0.007 | 0.020 | 0.017 | 0.028 | -0.029 | 0.048 | 0.015 |
| XNO | -0.042 | 0.018 | -0.005 | 0.010 | -0.042 | -0.026 | -0.017 | 0.027 | 0.012 | 1.000 | 0.002 | 0.026 | 0.011 | -0.014 | -0.018 | -0.022 | -0.025 | 0.029 | -0.009 | 0.030 | 0.011 | -0.007 | 0.023 | 0.001 |
| PHAROS | 0.065 | 0.021 | -0.025 | -0.008 | -0.022 | -0.006 | -0.006 | 0.018 | -0.006 | 0.002 | 1.000 | 0.003 | 0.008 | 0.022 | 0.014 | 0.009 | -0.025 | 0.027 | -0.005 | 0.002 | -0.004 | 0.024 | -0.013 | 0.068 |
| TURTLE | 0.039 | -0.013 | -0.029 | 0.035 | 0.016 | 0.009 | -0.026 | 0.041 | -0.015 | 0.026 | 0.003 | 1.000 | -0.019 | 0.041 | -0.006 | 0.025 | 0.019 | 0.013 | 0.004 | 0.011 | 0.002 | -0.039 | 0.020 | -0.002 |
| LUMIA | 0.013 | 0.042 | 0.033 | -0.018 | -0.000 | -0.060 | -0.027 | 0.004 | -0.022 | 0.011 | 0.008 | -0.019 | 1.000 | 0.001 | -0.011 | 0.007 | 0.043 | 0.014 | 0.039 | -0.019 | 0.041 | 0.033 | -0.007 | 0.005 |
| AMP | 0.061 | -0.023 | 0.012 | -0.001 | 0.011 | -0.019 | -0.004 | 0.054 | -0.008 | -0.014 | 0.022 | 0.041 | 0.001 | 1.000 | 0.009 | 0.005 | 0.012 | 0.014 | -0.011 | 0.007 | -0.034 | -0.049 | 0.013 | 0.044 |
| ESPORTS | 0.046 | -0.052 | 0.039 | 0.001 | -0.016 | -0.027 | -0.034 | 0.018 | 0.029 | -0.018 | 0.014 | -0.006 | -0.011 | 0.009 | 1.000 | -0.017 | 0.007 | -0.028 | -0.021 | -0.004 | -0.044 | 0.045 | -0.000 | 0.020 |
| HOME | 0.015 | -0.007 | -0.034 | -0.003 | 0.028 | 0.070 | 0.036 | 0.011 | -0.037 | -0.022 | 0.009 | 0.025 | 0.007 | 0.005 | -0.017 | 1.000 | -0.021 | 0.017 | -0.028 | -0.018 | 0.021 | -0.032 | -0.040 | 0.019 |
| US | 0.039 | -0.013 | 0.042 | 0.020 | 0.045 | -0.005 | -0.046 | 0.005 | 0.026 | -0.025 | -0.025 | 0.019 | 0.043 | 0.012 | 0.007 | -0.021 | 1.000 | -0.027 | -0.043 | 0.005 | -0.047 | 0.040 | -0.019 | -0.077 |
| AI | 0.079 | -0.022 | 0.000 | 0.029 | -0.022 | -0.040 | -0.000 | -0.005 | -0.007 | 0.029 | 0.027 | 0.013 | 0.014 | 0.014 | -0.028 | 0.017 | -0.027 | 1.000 | -0.006 | 0.045 | 0.035 | 0.008 | 0.024 | -0.020 |
| U | 0.047 | -0.021 | -0.019 | -0.058 | 0.038 | -0.013 | -0.011 | 0.037 | 0.020 | -0.009 | -0.005 | 0.004 | 0.039 | -0.011 | -0.021 | -0.028 | -0.043 | -0.006 | 1.000 | -0.006 | 0.005 | -0.004 | -0.018 | -0.000 |
| EVAA | 0.088 | -0.020 | -0.002 | -0.025 | 0.048 | -0.026 | 0.023 | -0.010 | 0.017 | 0.030 | 0.002 | 0.011 | -0.019 | 0.007 | -0.004 | -0.018 | 0.005 | 0.045 | -0.006 | 1.000 | 0.034 | -0.026 | 0.043 | 0.036 |
| SYN | -0.046 | 0.039 | -0.044 | -0.008 | 0.011 | -0.040 | -0.019 | -0.032 | 0.028 | 0.011 | -0.004 | 0.002 | 0.041 | -0.034 | -0.044 | 0.021 | -0.047 | 0.035 | 0.005 | 0.034 | 1.000 | -0.021 | 0.053 | -0.026 |
| JST | 0.020 | 0.065 | -0.032 | 0.011 | -0.047 | 0.024 | -0.028 | -0.002 | -0.029 | -0.007 | 0.024 | -0.039 | 0.033 | -0.049 | 0.045 | -0.032 | 0.040 | 0.008 | -0.004 | -0.026 | -0.021 | 1.000 | -0.020 | -0.005 |
| T | 0.070 | 0.029 | 0.022 | 0.037 | 0.048 | 0.041 | -0.005 | -0.033 | 0.048 | 0.023 | -0.013 | 0.020 | -0.007 | 0.013 | -0.000 | -0.040 | -0.019 | 0.024 | -0.018 | 0.043 | 0.053 | -0.020 | 1.000 | 0.006 |
| ENSO | 0.013 | 0.013 | 0.015 | -0.047 | 0.039 | 0.044 | 0.011 | 0.048 | 0.015 | 0.001 | 0.068 | -0.002 | 0.005 | 0.044 | 0.020 | 0.019 | -0.077 | -0.020 | -0.000 | 0.036 | -0.026 | -0.005 | 0.006 | 1.000 |

The complete 321 × 321 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| ZM | ZMUSDT | 59.1% | 63.9% | CRWDUSDT | 0.244 |
| ANKR | ANKRUSDT | 59.3% | 63.8% | QKCUSDT | 0.347 |
| SOPH | SOPHUSDT | 59.4% | 63.7% | BTCUSDT | 0.317 |
| JPM | JPMUSDT | 59.5% | 63.6% | SPYBUSDT | 0.193 |
| ZORA | ZORAUSDT | 59.8% | 63.4% | BTCUSDT | 0.285 |
| EGLD | EGLDUSDT | 59.9% | 63.3% | GUSDT | 0.376 |
| ASR | ASRUSDT | 59.9% | 63.3% | XAIUSDT | 0.250 |
| BANANAS31 | BANANAS31USDT | 59.9% | 63.3% | BTCUSDT | 0.364 |
| HANA | HANAUSDT | 60.0% | 63.2% | REQUSDT | -0.220 |
| RESOLV | RESOLVUSDT | 60.3% | 63.0% | OPGUSDT | 0.188 |
| MOCA | MOCAUSDT | 60.3% | 63.0% | BTCUSDT | 0.401 |
| SAGA | SAGAUSDT | 60.3% | 63.0% | XAIUSDT | 0.285 |
| QCOM | QCOMBUSDT | 60.5% | 62.9% | AVGOBUSDT | 0.361 |
| HEMI | HEMIUSDT | 60.5% | 62.9% | ACTUSDT | 0.201 |
| HOLO | HOLOUSDT | 60.5% | 62.8% | BTCUSDT | 0.315 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

