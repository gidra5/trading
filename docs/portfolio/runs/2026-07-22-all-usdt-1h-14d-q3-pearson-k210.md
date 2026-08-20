# Binance portfolio basis

Generated 2026-07-23T18:23:33.361Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-09T00:00Z through 2026-07-22T23:00Z
- Sampling: 1h log returns (336 samples over 14 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 671 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.2B | 30.7% |
| 2 | PROM | PROMUSDT (spot) | spot, usdm-futures | 100.0% | 0.001 | BTCUSDT | 100.0% | 2 | 277.7K | 216.2% |
| 3 | TAC | TACUSDT (usdm-futures) | usdm-futures | 100.0% | 0.004 | PROMUSDT | 100.0% | 1 | 12.8M | 326.8% |
| 4 | BTCDOM | BTCDOMUSDT (usdm-futures) | usdm-futures | 100.0% | 0.016 | BTCUSDT | 100.0% | 0 | 1.7M | 15.7% |
| 5 | B | BUSDT (usdm-futures) | usdm-futures | 100.0% | 0.017 | PROMUSDT | 100.0% | 1 | 30M | 412.9% |
| 6 | DODO | DODOUSDT (spot) | spot | 99.9% | 0.037 | TACUSDT | 100.0% | 1 | 4.1M | 330.5% |
| 7 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.9% | 0.042 | BTCDOMUSDT | 100.0% | 1 | 193.3M | 494.2% |
| 8 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 99.7% | 0.052 | BTCUSDT | 100.0% | 1 | 7M | 199.2% |
| 9 | XEC | XECUSDT (spot) | spot, usdm-futures | 99.6% | 0.054 | BUSDT | 100.0% | 2 | 2.8M | 225.6% |
| 10 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 99.5% | 0.059 | BTCDOMUSDT | 100.0% | 0 | 11.1M | 238.5% |
| 11 | O | OUSDT (usdm-futures) | usdm-futures | 99.4% | 0.071 | TACUSDT | 100.0% | 2 | 9M | 145.7% |
| 12 | HEI | HEIUSDT (spot) | spot, usdm-futures | 99.4% | 0.064 | BTCDOMUSDT | 100.0% | 1 | 1.9M | 154.2% |
| 13 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 99.3% | 0.081 | BTCUSDT | 100.0% | 0 | 3.8M | 218.9% |
| 14 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 99.2% | 0.066 | TACUSDT | 100.0% | 1 | 6.9M | 282.5% |
| 15 | U | UUSDT (spot) | spot | 99.1% | 0.086 | BTCDOMUSDT | 100.0% | 7 | 13.5M | 0.7% |
| 16 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 99.0% | 0.070 | DODOUSDT | 100.0% | 0 | 5.2M | 256.7% |
| 17 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 98.9% | 0.099 | TACUSDT | 100.0% | 1 | 8.1M | 315.6% |
| 18 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 98.6% | 0.082 | HEIUSDT | 100.0% | 0 | 2.6M | 142.9% |
| 19 | THE | THEUSDT (spot) | spot, usdm-futures | 98.5% | 0.097 | BTCUSDT | 100.0% | 2 | 783.2K | 148.1% |
| 20 | BTTC | BTTCUSDT (spot) | spot | 98.4% | 0.093 | PROMUSDT | 100.0% | 17 | 134.1K | 224.3% |
| 21 | RARE | RAREUSDT (spot) | spot, usdm-futures | 98.3% | 0.082 | HEIUSDT | 100.0% | 12 | 197.6K | 103.3% |
| 22 | XNO | XNOUSDT (spot) | spot | 98.2% | 0.100 | OUSDT | 100.0% | 3 | 43.7K | 330.9% |
| 23 | TUT | TUTUSDT (spot) | spot, usdm-futures | 98.1% | 0.098 | BTCUSDT | 100.0% | 1 | 514.3K | 114.4% |
| 24 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 97.9% | 0.096 | BTTCUSDT | 100.0% | 1 | 2M | 124.6% |
| 25 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 97.8% | 0.088 | BTCUSDT | 100.0% | 1 | 3.4M | 105.6% |
| 26 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 97.7% | 0.094 | PROMUSDT | 100.0% | 1 | 5M | 191.3% |
| 27 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 97.5% | 0.110 | UUSDT | 100.0% | 1 | 534.4K | 30.9% |
| 28 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 97.3% | 0.117 | BTCUSDT | 100.0% | 1 | 4.4M | 217.8% |
| 29 | MMT | MMTUSDT (spot) | spot, usdm-futures | 97.2% | 0.107 | TUTUSDT | 100.0% | 1 | 1.2M | 132.8% |
| 30 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 97.1% | 0.111 | BTCUSDT | 100.0% | 1 | 2.2M | 59.2% |
| 31 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 97.0% | 0.107 | OUSDT | 100.0% | 1 | 1.7M | 238.3% |
| 32 | BAR | BARUSDT (spot) | spot | 97.0% | 0.119 | DODOUSDT | 100.0% | 3 | 477.3K | 118.3% |
| 33 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 96.6% | 0.149 | BTCUSDT | 100.0% | 1 | 1M | 88.5% |
| 34 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 96.6% | 0.142 | TACUSDT | 100.0% | 1 | 27.7M | 306.8% |
| 35 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 96.2% | 0.106 | UUSDT | 100.0% | 1 | 5.2M | 213.2% |
| 36 | PORTO | PORTOUSDT (spot) | spot | 96.2% | 0.117 | XECUSDT | 100.0% | 2 | 547.5K | 253.0% |
| 37 | BR | BRUSDT (usdm-futures) | usdm-futures | 95.9% | 0.126 | BUSDT | 100.0% | 1 | 934.6K | 107.0% |
| 38 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 95.7% | 0.112 | BTCDOMUSDT | 100.0% | 1 | 5.8M | 239.1% |
| 39 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 95.5% | 0.153 | PHAROSUSDT | 100.0% | 1 | 2.5M | 186.4% |
| 40 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 95.3% | 0.109 | 币安人生USDT | 100.0% | 1 | 1.6M | 69.1% |
| 41 | DCR | DCRUSDT (spot) | spot | 95.2% | 0.181 | BTCUSDT | 100.0% | 2 | 370.8K | 169.6% |
| 42 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 95.1% | 0.129 | ZESTUSDT | 100.0% | 1 | 30.8M | 265.0% |
| 43 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 94.9% | 0.122 | ALCHUSDT | 100.0% | 1 | 2.6M | 229.0% |
| 44 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 94.8% | 0.132 | NFLXUSDT | 100.0% | 1 | 17.7M | 26.1% |
| 45 | ONE | ONEUSDT (spot) | spot, usdm-futures | 94.7% | 0.198 | BTCUSDT | 100.0% | 5 | 214.4K | 196.6% |
| 46 | ON | ONUSDT (usdm-futures) | usdm-futures | 94.3% | 0.112 | PROMUSDT | 100.0% | 0 | 2.8M | 263.9% |
| 47 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 94.2% | 0.118 | HEIUSDT | 100.0% | 1 | 2.2M | 139.4% |
| 48 | SENT | SENTUSDT (spot) | spot, usdm-futures | 93.9% | 0.121 | BTCDOMUSDT | 100.0% | 1 | 1.6M | 132.2% |
| 49 | ACE | ACEUSDT (spot) | spot, usdm-futures | 93.7% | 0.112 | AVAAIUSDT | 100.0% | 2 | 166.3K | 260.2% |
| 50 | GLMR | GLMRUSDT (spot) | spot | 93.5% | 0.140 | BRUSDT | 100.0% | 9 | 387.2K | 88.0% |
| 51 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 93.2% | 0.164 | BTCDOMUSDT | 100.0% | 2 | 600.2K | 163.7% |
| 52 | TTWO | TTWOUSDT (usdm-futures) | usdm-futures | 93.1% | 0.128 | TUTUSDT | 100.0% | 1 | 208.1K | 34.2% |
| 53 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 93.0% | 0.150 | ANTHROPICUSDT | 100.0% | 1 | 1.8M | 138.1% |
| 54 | H | HUSDT (usdm-futures) | usdm-futures | 92.9% | 0.157 | VELVETUSDT | 100.0% | 1 | 6.6M | 120.6% |
| 55 | SYN | SYNUSDT (spot) | spot, usdm-futures | 92.6% | 0.139 | BTCDOMUSDT | 100.0% | 0 | 6.9M | 273.6% |
| 56 | OGN | OGNUSDT (spot) | spot, usdm-futures | 92.5% | 0.169 | BTCUSDT | 100.0% | 1 | 788.2K | 112.1% |
| 57 | RIF | RIFUSDT (spot) | spot, usdm-futures | 92.4% | 0.179 | 币安人生USDT | 100.0% | 1 | 1.7M | 249.8% |
| 58 | B2 | B2USDT (usdm-futures) | usdm-futures | 92.4% | 0.154 | UUSDT | 100.0% | 1 | 889.8K | 125.7% |
| 59 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 92.0% | 0.124 | UUSDT | 100.0% | 2 | 2M | 169.9% |
| 60 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 91.8% | 0.183 | 币安人生USDT | 100.0% | 1 | 642.1K | 55.2% |
| 61 | PIVX | PIVXUSDT (spot) | spot | 91.5% | 0.246 | BLESSUSDT | 100.0% | 2 | 291.3K | 135.6% |
| 62 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 91.4% | 0.124 | BTCUSDT | 100.0% | 3 | 833.3K | 129.1% |
| 63 | AT | ATUSDT (spot) | spot, usdm-futures | 91.3% | 0.184 | BTCUSDT | 100.0% | 1 | 259.7K | 83.5% |
| 64 | DGB | DGBUSDT (spot) | spot | 91.1% | 0.173 | BTTCUSDT | 100.0% | 3 | 187.1K | 236.9% |
| 65 | GPS | GPSUSDT (spot) | spot, usdm-futures | 91.0% | 0.156 | BTCUSDT | 100.0% | 2 | 351.7K | 85.4% |
| 66 | SUN | SUNUSDT (spot) | spot, usdm-futures | 90.8% | 0.194 | THEUSDT | 100.0% | 3 | 602.6K | 24.1% |
| 67 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 90.6% | 0.132 | BUSDT | 100.0% | 2 | 2.4M | 160.5% |
| 68 | AWE | AWEUSDT (spot) | spot, usdm-futures | 90.3% | 0.114 | AAPLUSDT | 100.0% | 1 | 322.7K | 103.8% |
| 69 | BAN | BANUSDT (usdm-futures) | usdm-futures | 90.2% | 0.161 | SENTUSDT | 100.0% | 2 | 1.3M | 98.2% |
| 70 | QI | QIUSDT (spot) | spot | 90.0% | 0.164 | BTCDOMUSDT | 100.0% | 10 | 124.9K | 99.5% |
| 71 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 89.7% | 0.157 | ACEUSDT | 100.0% | 1 | 2.7M | 124.9% |
| 72 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 89.6% | 0.123 | BRUSDT | 100.0% | 1 | 1.4M | 98.9% |
| 73 | LA | LAUSDT (spot) | spot, usdm-futures | 89.1% | 0.173 | EPICUSDT | 100.0% | 2 | 481.7K | 134.7% |
| 74 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 89.0% | 0.135 | TRADOORUSDT | 100.0% | 1 | 6.4M | 133.1% |
| 75 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 88.7% | 0.189 | TAGUSDT | 100.0% | 1 | 2.3M | 74.9% |
| 76 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 88.5% | 0.163 | PROMUSDT | 100.0% | 1 | 270.7K | 44.3% |
| 77 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 88.3% | 0.184 | SENTUSDT | 100.0% | 1 | 9.7M | 187.7% |
| 78 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 88.1% | 0.299 | BTCUSDT | 100.0% | 2 | 897.2K | 130.5% |
| 79 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 88.0% | 0.204 | BTCUSDT | 100.0% | 2 | 822K | 98.0% |
| 80 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 87.6% | 0.207 | BTCUSDT | 100.0% | 1 | 1.4M | 111.1% |
| 81 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 87.2% | 0.174 | RIFUSDT | 100.0% | 2 | 217.8K | 33.8% |
| 82 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 87.0% | 0.275 | BLESSUSDT | 100.0% | 1 | 917.9K | 138.9% |
| 83 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 87.0% | 0.185 | BTCUSDT | 100.0% | 2 | 1.6M | 103.7% |
| 84 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 86.7% | 0.273 | RIFUSDT | 100.0% | 1 | 25M | 467.6% |
| 85 | ATM | ATMUSDT (spot) | spot | 86.4% | 0.192 | GLMRUSDT | 100.0% | 1 | 1.3M | 259.5% |
| 86 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 86.4% | 0.158 | AGLDUSDT | 100.0% | 1 | 20M | 183.3% |
| 87 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 86.3% | 0.201 | BTCUSDT | 100.0% | 5 | 315.4K | 92.0% |
| 88 | KGST | KGSTUSDT (spot) | spot | 86.1% | 0.245 | SENTUSDT | 100.0% | 6 | 367.9K | 6.6% |
| 89 | YB | YBUSDT (spot) | spot, usdm-futures | 86.0% | 0.233 | BLUAIUSDT | 100.0% | 2 | 347.9K | 119.1% |
| 90 | SXT | SXTUSDT (spot) | spot, usdm-futures | 85.9% | 0.129 | DODOUSDT | 100.0% | 1 | 2.9M | 204.6% |
| 91 | QUICK | QUICKUSDT (spot) | spot | 85.7% | 0.167 | DCRUSDT | 100.0% | 2 | 54.4K | 69.6% |
| 92 | JST | JSTUSDT (spot) | spot, usdm-futures | 85.3% | 0.161 | BILLUSDT | 100.0% | 0 | 2.9M | 46.0% |
| 93 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 85.1% | 0.158 | AVAAIUSDT | 100.0% | 1 | 84.9M | 566.7% |
| 94 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 85.0% | 0.173 | TTWOUSDT | 100.0% | 1 | 3M | 153.0% |
| 95 | M | MUSDT (usdm-futures) | usdm-futures | 84.5% | 0.182 | BILLUSDT | 100.0% | 1 | 3.2M | 162.4% |
| 96 | AIN | AINUSDT (usdm-futures) | usdm-futures | 84.5% | 0.147 | THEUSDT | 100.0% | 1 | 909.1K | 130.5% |
| 97 | NMR | NMRUSDT (spot) | spot, usdm-futures | 84.5% | 0.172 | BTCUSDT | 100.0% | 2 | 219.1K | 56.3% |
| 98 | LAB | LABUSDT (usdm-futures) | usdm-futures | 83.8% | 0.177 | TACUSDT | 100.0% | 0 | 248.9M | 455.7% |
| 99 | BAS | BASUSDT (usdm-futures) | usdm-futures | 83.8% | 0.219 | BILLUSDT | 100.0% | 1 | 2.8M | 140.1% |
| 100 | AUDIO | AUDIOUSDT (spot) | spot | 83.5% | 0.185 | BANUSDT | 100.0% | 2 | 370K | 64.8% |
| 101 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 83.4% | 0.260 | BTCUSDT | 100.0% | 5 | 676.8K | 65.1% |
| 102 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 83.3% | 0.129 | DEXEUSDT | 100.0% | 1 | 2.8M | 167.8% |
| 103 | VANA | VANAUSDT (spot) | spot, usdm-futures | 83.1% | 0.152 | YBUSDT | 100.0% | 2 | 1.1M | 82.8% |
| 104 | OPN | OPNUSDT (spot) | spot, usdm-futures | 82.9% | 0.176 | OGNUSDT | 100.0% | 3 | 18.1M | 111.7% |
| 105 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 82.4% | 0.297 | THEUSDT | 100.0% | 1 | 836.8K | 83.0% |
| 106 | TREE | TREEUSDT (spot) | spot, usdm-futures | 82.3% | 0.177 | BTCDOMUSDT | 100.0% | 3 | 6.8M | 165.2% |
| 107 | CATI | CATIUSDT (spot) | spot, usdm-futures | 82.2% | 0.185 | BTCUSDT | 100.0% | 2 | 325K | 92.8% |
| 108 | BEL | BELUSDT (spot) | spot, usdm-futures | 81.9% | 0.154 | KGSTUSDT | 100.0% | 1 | 723.1K | 93.5% |
| 109 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 81.8% | 0.175 | BTCDOMUSDT | 100.0% | 1 | 3.1M | 131.4% |
| 110 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 81.5% | 0.175 | BLUAIUSDT | 100.0% | 1 | 1.8M | 135.0% |
| 111 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 81.3% | 0.166 | BTCDOMUSDT | 100.0% | 1 | 36M | 120.1% |
| 112 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 81.1% | 0.170 | BTWUSDT | 100.0% | 4 | 1.2M | 183.9% |
| 113 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 81.0% | 0.186 | TREEUSDT | 100.0% | 2 | 1.4M | 83.0% |
| 114 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 80.7% | 0.207 | AIOUSDT | 100.0% | 2 | 523.6K | 99.3% |
| 115 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 80.2% | 0.188 | GLMRUSDT | 100.0% | 1 | 1.2M | 173.4% |
| 116 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 79.8% | 0.165 | PIVXUSDT | 100.0% | 2 | 12.1M | 192.8% |
| 117 | ID | IDUSDT (spot) | spot, usdm-futures | 79.7% | 0.211 | BTCUSDT | 100.0% | 2 | 690.1K | 84.2% |
| 118 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 79.6% | 0.294 | BTCUSDT | 100.0% | 1 | 17.7M | 62.6% |
| 119 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 79.5% | 0.179 | HOMEUSDT | 100.0% | 1 | 1.7M | 87.7% |
| 120 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 79.1% | 0.260 | BTCUSDT | 100.0% | 1 | 1.5M | 86.3% |
| 121 | T | TUSDT (spot) | spot, usdm-futures | 79.0% | 0.300 | ANKRUSDT | 100.0% | 4 | 2M | 204.8% |
| 122 | KITE | KITEUSDT (spot) | spot, usdm-futures | 78.9% | 0.185 | DCRUSDT | 100.0% | 1 | 11.7M | 110.0% |
| 123 | GNO | GNOUSDT (spot) | spot | 78.8% | 0.208 | AKEUSDT | 100.0% | 2 | 128.1K | 74.9% |
| 124 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 78.6% | 0.189 | BTCUSDT | 100.0% | 2 | 732.3K | 102.2% |
| 125 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 78.3% | 0.158 | TACUSDT | 100.0% | 0 | 13.7M | 153.2% |
| 126 | SCRT | SCRTUSDT (spot) | spot, usdm-futures | 78.0% | 0.192 | VANRYUSDT | 100.0% | 2 | 225.9K | 83.7% |
| 127 | CC | CCUSDT (usdm-futures) | usdm-futures | 77.7% | 0.147 | DATAIPUSDT | 100.0% | 1 | 3.5M | 56.6% |
| 128 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 77.6% | 0.174 | DEXEUSDT | 100.0% | 1 | 661.8K | 99.9% |
| 129 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 77.2% | 0.249 | BANUSDT | 100.0% | 2 | 1.2M | 97.9% |
| 130 | TA | TAUSDT (usdm-futures) | usdm-futures | 77.1% | 0.162 | ARXUSDT | 100.0% | 1 | 1.1M | 66.8% |
| 131 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 76.8% | 0.192 | AIGENSYNUSDT | 100.0% | 1 | 1.1M | 92.8% |
| 132 | FTT | FTTUSDT (spot) | spot | 76.7% | 0.317 | BTCUSDT | 100.0% | 2 | 191.6K | 85.1% |
| 133 | Q | QUSDT (usdm-futures) | usdm-futures | 76.7% | 0.177 | EDGEUSDT | 100.0% | 1 | 1.9M | 115.3% |
| 134 | UB | UBUSDT (usdm-futures) | usdm-futures | 76.5% | 0.210 | LABUSDT | 100.0% | 1 | 14.3M | 192.6% |
| 135 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 76.2% | 0.278 | BTCUSDT | 100.0% | 4 | 2M | 78.3% |
| 136 | BX | BXUSDT (usdm-futures) | usdm-futures | 75.9% | 0.175 | QIUSDT | 100.0% | 1 | 831K | 47.5% |
| 137 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 75.8% | 0.176 | AKEUSDT | 100.0% | 3 | 4.6M | 146.7% |
| 138 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 75.5% | 0.165 | NFLXUSDT | 100.0% | 6 | 385.4K | 62.6% |
| 139 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 75.1% | 0.227 | BTCUSDT | 100.0% | 2 | 990.7K | 71.0% |
| 140 | MET | METUSDT (spot) | spot, usdm-futures | 75.0% | 0.243 | BTCUSDT | 100.0% | 1 | 606.4K | 113.4% |
| 141 | MBL | MBLUSDT (spot) | spot | 74.9% | 0.258 | BTCUSDT | 100.0% | 3 | 448.6K | 51.2% |
| 142 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 74.5% | 0.182 | QIUSDT | 100.0% | 1 | 2.1M | 109.9% |
| 143 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 74.1% | 0.191 | BTCUSDT | 100.0% | 10 | 180.8K | 27.7% |
| 144 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 74.0% | 0.243 | VANAUSDT | 100.0% | 4 | 719.3K | 97.5% |
| 145 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 74.0% | 0.191 | PROMUSDT | 100.0% | 2 | 926.2K | 149.5% |
| 146 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 73.8% | 0.220 | SPELLUSDT | 100.0% | 0 | 150M | 956.9% |
| 147 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 73.4% | 0.251 | SENTUSDT | 100.0% | 1 | 1.1M | 101.8% |
| 148 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 73.2% | 0.176 | AIOUSDT | 100.0% | 1 | 8.5M | 186.3% |
| 149 | WET | WETUSDT (usdm-futures) | usdm-futures | 73.0% | 0.233 | BTCUSDT | 100.0% | 1 | 1.4M | 86.8% |
| 150 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 72.9% | 0.183 | PRLUSDT | 100.0% | 1 | 13M | 230.9% |
| 151 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 72.7% | 0.243 | PIVXUSDT | 100.0% | 1 | 2.4M | 132.6% |
| 152 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 72.5% | 0.224 | LIGHTUSDT | 100.0% | 1 | 21.3M | 140.4% |
| 153 | WIN | WINUSDT (spot) | spot | 72.4% | 0.289 | BTCUSDT | 100.0% | 2 | 98.2K | 50.9% |
| 154 | US | USUSDT (usdm-futures) | usdm-futures | 72.0% | 0.209 | UBUSDT | 100.0% | 0 | 44.5M | 313.2% |
| 155 | APR | APRUSDT (usdm-futures) | usdm-futures | 71.7% | 0.310 | AIOTUSDT | 100.0% | 2 | 3.3M | 106.3% |
| 156 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 71.3% | 0.205 | AGLDUSDT | 100.0% | 1 | 19.9M | 182.0% |
| 157 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 70.9% | 0.277 | BLESSUSDT | 100.0% | 3 | 2.1M | 94.7% |
| 158 | PYR | PYRUSDT (spot) | spot | 70.7% | 0.330 | RAREUSDT | 100.0% | 3 | 1.2M | 273.7% |
| 159 | GNS | GNSUSDT (spot) | spot | 70.6% | 0.248 | ACEUSDT | 100.0% | 4 | 51.5K | 42.0% |
| 160 | MASK | MASKUSDT (spot) | spot, usdm-futures | 70.3% | 0.343 | BTCUSDT | 100.0% | 5 | 194.3K | 49.9% |
| 161 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 70.3% | 0.225 | BTCUSDT | 100.0% | 2 | 378.6K | 86.0% |
| 162 | MITO | MITOUSDT (spot) | spot, usdm-futures | 69.8% | 0.278 | THEUSDT | 100.0% | 2 | 573.8K | 106.1% |
| 163 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 69.6% | 0.235 | MANTRAUSDT | 100.0% | 2 | 295.6K | 32.2% |
| 164 | A | AUSDT (spot) | spot, usdm-futures | 69.3% | 0.230 | BTCUSDT | 100.0% | 2 | 343.3K | 78.0% |
| 165 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 69.3% | 0.283 | DCRUSDT | 100.0% | 2 | 1.8M | 237.3% |
| 166 | KNC | KNCUSDT (spot) | spot, usdm-futures | 68.8% | 0.247 | TAUSDT | 100.0% | 3 | 71.7K | 77.6% |
| 167 | FF | FFUSDT (spot) | spot, usdm-futures | 68.6% | 0.242 | QIUSDT | 100.0% | 1 | 1.2M | 77.5% |
| 168 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 68.5% | 0.253 | BTCUSDT | 100.0% | 4 | 467.2K | 60.5% |
| 169 | MAV | MAVUSDT (spot) | spot, usdm-futures | 68.3% | 0.361 | BTCUSDT | 100.0% | 2 | 171.6K | 74.2% |
| 170 | PHA | PHAUSDT (spot) | spot, usdm-futures | 67.8% | 0.321 | BTCUSDT | 100.0% | 4 | 517.6K | 85.8% |
| 171 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 67.6% | 0.241 | BTCUSDT | 100.0% | 1 | 1.8M | 62.7% |
| 172 | SOLV | SOLVUSDT (spot) | spot, usdm-futures | 66.9% | 0.355 | ARIAUSDT | 100.0% | 6 | 161.7K | 99.3% |
| 173 | ASTS | ASTSUSDT (usdm-futures) | usdm-futures | 66.8% | 0.365 | SPCXBUSDT | 100.0% | 1 | 2.4M | 117.2% |
| 174 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 66.6% | 0.210 | PROMUSDT | 100.0% | 1 | 2M | 92.9% |
| 175 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 66.5% | 0.241 | IDUSDT | 100.0% | 2 | 604.3K | 69.0% |
| 176 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 66.3% | 0.271 | 币安人生USDT | 100.0% | 1 | 3.6M | 145.8% |
| 177 | HIMS | HIMSUSDT (usdm-futures) | usdm-futures | 65.9% | 0.281 | BTCUSDT | 100.0% | 2 | 831.8K | 61.9% |
| 178 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 65.7% | 0.209 | PROMPTUSDT | 100.0% | 1 | 1.2M | 92.0% |
| 179 | 4 | 4USDT (usdm-futures) | usdm-futures | 65.3% | 0.220 | TNSRUSDT | 100.0% | 1 | 1.8M | 69.5% |
| 180 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 65.2% | 0.271 | BTCUSDT | 100.0% | 2 | 4.5M | 82.7% |
| 181 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 64.8% | 0.161 | ONUSDT | 100.0% | 1 | 967.9K | 84.6% |
| 182 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 64.6% | 0.198 | BTCUSDT | 100.0% | 3 | 2.4M | 38.2% |
| 183 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 64.2% | 0.331 | MASKUSDT | 100.0% | 0 | 913.2K | 65.9% |
| 184 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 64.0% | 0.270 | PORTOUSDT | 100.0% | 1 | 1.6M | 88.7% |
| 185 | QKC | QKCUSDT (spot) | spot | 63.8% | 0.300 | BTCUSDT | 100.0% | 2 | 80.3K | 46.5% |
| 186 | ADX | ADXUSDT (spot) | spot | 63.7% | 0.400 | BTCUSDT | 100.0% | 3 | 163.7K | 54.3% |
| 187 | HOT | HOTUSDT (spot) | spot, usdm-futures | 63.5% | 0.269 | BTCUSDT | 100.0% | 3 | 246.9K | 109.7% |
| 188 | BTR | BTRUSDT (usdm-futures) | usdm-futures | 63.4% | 0.377 | BTCUSDT | 100.0% | 2 | 764.3K | 66.1% |
| 189 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 63.3% | 0.197 | AUSDT | 100.0% | 1 | 1M | 82.9% |
| 190 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 62.8% | 0.403 | BTCUSDT | 100.0% | 2 | 572.7K | 61.1% |
| 191 | DIS | DISUSDT (usdm-futures) | usdm-futures | 62.7% | 0.195 | AAPLUSDT | 100.0% | 2 | 224.5K | 26.4% |
| 192 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 62.4% | 0.207 | VANRYUSDT | 100.0% | 0 | 8.9M | 231.6% |
| 193 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 62.1% | 0.217 | GLMRUSDT | 100.0% | 2 | 792.2K | 82.3% |
| 194 | XAN | XANUSDT (usdm-futures) | usdm-futures | 61.8% | 0.177 | BLESSUSDT | 100.0% | 0 | 2.8M | 150.3% |
| 195 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 61.8% | 0.269 | LABUSDT | 100.0% | 2 | 7.5M | 133.7% |
| 196 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 61.6% | 0.267 | MITOUSDT | 100.0% | 1 | 5M | 143.5% |
| 197 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 61.1% | 0.278 | TOSHIUSDT | 100.0% | 1 | 159.7K | 73.6% |
| 198 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 60.9% | 0.204 | BULLAUSDT | 100.0% | 0 | 1.4M | 103.4% |
| 199 | PYTH | PYTHUSDT (spot) | spot, usdm-futures | 60.6% | 0.268 | BTCUSDT | 100.0% | 1 | 1.9M | 81.4% |
| 200 | MOCA | MOCAUSDT (usdm-futures) | usdm-futures | 60.5% | 0.353 | BTCUSDT | 100.0% | 2 | 431.8K | 63.3% |
| 201 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 60.0% | 0.295 | BTCUSDT | 100.0% | 1 | 1.9M | 84.3% |
| 202 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 59.7% | 0.247 | KNCUSDT | 100.0% | 1 | 397.8K | 76.7% |
| 203 | ACX | ACXUSDT (spot) | spot, usdm-futures | 59.5% | 0.363 | BTCUSDT | 100.0% | 2 | 73.6K | 33.4% |
| 204 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 59.3% | 0.351 | BTCUSDT | 100.0% | 2 | 165.9K | 57.8% |
| 205 | LIT | LITUSDT (usdm-futures) | usdm-futures | 59.0% | 0.448 | BTCUSDT | 100.0% | 1 | 50.7M | 119.1% |
| 206 | OPG | OPGUSDT (spot) | spot, usdm-futures | 58.2% | 0.308 | PROMPTUSDT | 100.0% | 2 | 1.2M | 86.1% |
| 207 | POL | POLUSDT (spot) | spot, usdm-futures | 57.9% | 0.252 | BILLUSDT | 100.0% | 1 | 2.6M | 36.4% |
| 208 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 57.7% | 0.272 | ACEUSDT | 100.0% | 1 | 1.6M | 201.6% |
| 209 | EWJ | EWJUSDT (usdm-futures) | usdm-futures | 57.3% | 0.316 | BTCUSDT | 100.0% | 2 | 914.1K | 24.0% |
| 210 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 57.3% | 0.210 | ESPORTSUSDT | 100.0% | 1 | 1.8M | 120.5% |

## Diagnostics

- Basis size selected: 210
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.055
- Maximum pairwise absolute correlation: 0.448
- Mean whole-market projection R²: 84.0%
- Median whole-market projection R²: 80.0%
- 10th-percentile whole-market projection R²: 70.6%
- Minimum whole-market projection R²: 67.5%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.5% | 9.8% | 0.3% | 0.0% |
| 5 | 19.4% | 13.2% | 1.8% | 0.2% |
| 10 | 21.5% | 15.6% | 3.3% | 1.2% |
| 15 | 23.7% | 17.5% | 4.9% | 2.0% |
| 20 | 25.8% | 19.0% | 6.6% | 3.4% |
| 25 | 27.8% | 21.1% | 8.3% | 4.6% |
| 30 | 29.7% | 22.5% | 9.8% | 6.0% |
| 35 | 31.7% | 24.5% | 11.8% | 7.4% |
| 40 | 33.8% | 26.7% | 13.4% | 9.3% |
| 45 | 36.2% | 29.1% | 15.9% | 11.0% |
| 50 | 38.2% | 30.9% | 17.6% | 13.1% |
| 55 | 40.2% | 32.7% | 19.5% | 14.5% |
| 60 | 42.0% | 34.8% | 21.4% | 16.3% |
| 65 | 43.7% | 36.5% | 23.0% | 17.6% |
| 70 | 45.6% | 38.1% | 24.3% | 19.5% |
| 75 | 47.4% | 40.1% | 26.4% | 21.6% |
| 80 | 49.1% | 41.7% | 28.2% | 23.9% |
| 85 | 50.8% | 43.5% | 29.6% | 25.4% |
| 90 | 52.4% | 45.0% | 31.5% | 26.6% |
| 95 | 54.1% | 46.8% | 33.2% | 28.6% |
| 100 | 55.8% | 48.9% | 35.2% | 30.5% |
| 105 | 57.3% | 50.3% | 36.8% | 32.3% |
| 110 | 58.9% | 52.0% | 38.9% | 33.9% |
| 115 | 60.5% | 53.6% | 40.4% | 36.3% |
| 120 | 62.0% | 55.5% | 42.0% | 37.5% |
| 125 | 63.4% | 56.7% | 43.5% | 39.2% |
| 130 | 64.8% | 58.1% | 45.5% | 41.0% |
| 135 | 66.2% | 59.5% | 46.9% | 42.4% |
| 140 | 67.8% | 61.7% | 48.9% | 43.8% |
| 145 | 69.1% | 63.0% | 50.4% | 45.5% |
| 150 | 70.3% | 64.4% | 52.1% | 47.1% |
| 155 | 71.5% | 65.3% | 53.5% | 49.1% |
| 160 | 72.8% | 66.6% | 54.9% | 50.6% |
| 165 | 74.1% | 68.1% | 56.9% | 52.7% |
| 170 | 75.4% | 69.6% | 58.5% | 54.4% |
| 175 | 76.5% | 70.7% | 60.1% | 56.0% |
| 180 | 77.7% | 72.1% | 61.7% | 58.0% |
| 185 | 78.7% | 73.8% | 63.2% | 59.4% |
| 190 | 79.8% | 74.8% | 64.5% | 60.7% |
| 195 | 80.8% | 76.1% | 66.2% | 62.0% |
| 200 | 81.9% | 77.4% | 67.7% | 64.0% |
| 205 | 82.9% | 78.3% | 69.1% | 66.1% |
| 210 | 84.0% | 80.0% | 70.6% | 67.5% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | PROM | TAC | BTCDOM | B | DODO | AKE | ALLO | XEC | MAGMA | O | HEI | BLESS | SKYAI | U | AVAAI | TAG | AIO | THE | BTTC | RARE | XNO | TUT | ZEST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.001 | 0.003 | 0.016 | 0.009 | 0.019 | 0.007 | 0.052 | 0.037 | 0.029 | 0.023 | -0.019 | 0.081 | 0.022 | 0.062 | 0.061 | 0.051 | 0.036 | 0.097 | 0.048 | 0.081 | -0.041 | 0.098 | -0.006 |
| PROM | -0.001 | 1.000 | 0.004 | -0.006 | -0.017 | -0.004 | -0.020 | -0.006 | 0.019 | -0.038 | 0.042 | -0.027 | 0.046 | 0.021 | -0.008 | -0.029 | -0.022 | -0.013 | -0.031 | -0.093 | -0.014 | 0.057 | 0.051 | 0.092 |
| TAC | 0.003 | 0.004 | 1.000 | -0.005 | -0.005 | 0.037 | -0.006 | 0.016 | 0.009 | -0.019 | 0.071 | 0.037 | -0.001 | 0.066 | -0.034 | -0.033 | 0.099 | 0.018 | 0.024 | -0.032 | -0.024 | 0.040 | -0.036 | 0.003 |
| BTCDOM | 0.016 | -0.006 | -0.005 | 1.000 | -0.013 | 0.014 | -0.042 | -0.016 | -0.013 | -0.059 | -0.036 | -0.064 | -0.056 | -0.005 | 0.086 | -0.037 | -0.006 | -0.022 | -0.053 | -0.020 | -0.038 | 0.073 | -0.064 | -0.070 |
| B | 0.009 | -0.017 | -0.005 | -0.013 | 1.000 | -0.013 | 0.017 | 0.031 | 0.054 | -0.018 | -0.017 | -0.044 | 0.004 | 0.034 | -0.036 | -0.026 | 0.030 | 0.077 | -0.025 | 0.017 | 0.044 | 0.028 | 0.019 | 0.004 |
| DODO | 0.019 | -0.004 | 0.037 | 0.014 | -0.013 | 1.000 | -0.020 | -0.014 | -0.031 | 0.031 | 0.043 | -0.039 | -0.004 | 0.022 | 0.026 | -0.070 | 0.002 | -0.014 | 0.031 | -0.022 | -0.011 | 0.006 | 0.023 | -0.002 |
| AKE | 0.007 | -0.020 | -0.006 | -0.042 | 0.017 | -0.020 | 1.000 | -0.023 | 0.049 | 0.001 | 0.021 | 0.022 | 0.016 | -0.059 | -0.010 | -0.008 | 0.013 | 0.016 | -0.007 | -0.035 | -0.050 | -0.014 | 0.038 | -0.022 |
| ALLO | 0.052 | -0.006 | 0.016 | -0.016 | 0.031 | -0.014 | -0.023 | 1.000 | -0.002 | 0.037 | 0.018 | 0.037 | -0.025 | -0.059 | 0.047 | -0.020 | -0.052 | -0.069 | 0.010 | -0.005 | 0.012 | 0.021 | -0.028 | 0.005 |
| XEC | 0.037 | 0.019 | 0.009 | -0.013 | 0.054 | -0.031 | 0.049 | -0.002 | 1.000 | -0.022 | 0.018 | 0.005 | -0.001 | -0.005 | -0.009 | -0.032 | 0.002 | 0.000 | 0.008 | -0.033 | -0.074 | 0.024 | -0.013 | 0.025 |
| MAGMA | 0.029 | -0.038 | -0.019 | -0.059 | -0.018 | 0.031 | 0.001 | 0.037 | -0.022 | 1.000 | 0.031 | -0.019 | -0.024 | -0.033 | -0.008 | 0.025 | -0.005 | 0.044 | 0.042 | -0.046 | 0.008 | -0.022 | 0.042 | -0.077 |
| O | 0.023 | 0.042 | 0.071 | -0.036 | -0.017 | 0.043 | 0.021 | 0.018 | 0.018 | 0.031 | 1.000 | -0.011 | 0.002 | 0.020 | -0.011 | -0.038 | 0.044 | 0.022 | -0.034 | 0.023 | 0.037 | 0.100 | -0.044 | 0.045 |
| HEI | -0.019 | -0.027 | 0.037 | -0.064 | -0.044 | -0.039 | 0.022 | 0.037 | 0.005 | -0.019 | -0.011 | 1.000 | 0.001 | 0.002 | 0.006 | 0.039 | -0.033 | 0.082 | 0.072 | 0.064 | 0.082 | 0.038 | 0.045 | 0.079 |
| BLESS | 0.081 | 0.046 | -0.001 | -0.056 | 0.004 | -0.004 | 0.016 | -0.025 | -0.001 | -0.024 | 0.002 | 0.001 | 1.000 | 0.037 | -0.005 | 0.019 | 0.041 | 0.018 | 0.031 | 0.023 | -0.007 | -0.035 | -0.014 | 0.014 |
| SKYAI | 0.022 | 0.021 | 0.066 | -0.005 | 0.034 | 0.022 | -0.059 | -0.059 | -0.005 | -0.033 | 0.020 | 0.002 | 0.037 | 1.000 | -0.043 | 0.043 | -0.006 | 0.025 | 0.019 | 0.007 | 0.041 | -0.039 | -0.049 | -0.015 |
| U | 0.062 | -0.008 | -0.034 | 0.086 | -0.036 | 0.026 | -0.010 | 0.047 | -0.009 | -0.008 | -0.011 | 0.006 | -0.005 | -0.043 | 1.000 | -0.001 | 0.006 | -0.020 | -0.037 | -0.059 | -0.023 | 0.060 | -0.020 | -0.053 |
| AVAAI | 0.061 | -0.029 | -0.033 | -0.037 | -0.026 | -0.070 | -0.008 | -0.020 | -0.032 | 0.025 | -0.038 | 0.039 | 0.019 | 0.043 | -0.001 | 1.000 | -0.009 | -0.015 | -0.031 | -0.017 | 0.031 | -0.017 | -0.009 | 0.012 |
| TAG | 0.051 | -0.022 | 0.099 | -0.006 | 0.030 | 0.002 | 0.013 | -0.052 | 0.002 | -0.005 | 0.044 | -0.033 | 0.041 | -0.006 | 0.006 | -0.009 | 1.000 | 0.052 | 0.000 | 0.026 | -0.051 | 0.013 | 0.018 | -0.039 |
| AIO | 0.036 | -0.013 | 0.018 | -0.022 | 0.077 | -0.014 | 0.016 | -0.069 | 0.000 | 0.044 | 0.022 | 0.082 | 0.018 | 0.025 | -0.020 | -0.015 | 0.052 | 1.000 | -0.034 | -0.018 | 0.019 | 0.055 | 0.047 | 0.017 |
| THE | 0.097 | -0.031 | 0.024 | -0.053 | -0.025 | 0.031 | -0.007 | 0.010 | 0.008 | 0.042 | -0.034 | 0.072 | 0.031 | 0.019 | -0.037 | -0.031 | 0.000 | -0.034 | 1.000 | -0.025 | 0.039 | -0.028 | -0.025 | -0.004 |
| BTTC | 0.048 | -0.093 | -0.032 | -0.020 | 0.017 | -0.022 | -0.035 | -0.005 | -0.033 | -0.046 | 0.023 | 0.064 | 0.023 | 0.007 | -0.059 | -0.017 | 0.026 | -0.018 | -0.025 | 1.000 | 0.022 | -0.013 | 0.025 | 0.096 |
| RARE | 0.081 | -0.014 | -0.024 | -0.038 | 0.044 | -0.011 | -0.050 | 0.012 | -0.074 | 0.008 | 0.037 | 0.082 | -0.007 | 0.041 | -0.023 | 0.031 | -0.051 | 0.019 | 0.039 | 0.022 | 1.000 | 0.003 | 0.029 | 0.024 |
| XNO | -0.041 | 0.057 | 0.040 | 0.073 | 0.028 | 0.006 | -0.014 | 0.021 | 0.024 | -0.022 | 0.100 | 0.038 | -0.035 | -0.039 | 0.060 | -0.017 | 0.013 | 0.055 | -0.028 | -0.013 | 0.003 | 1.000 | 0.038 | 0.037 |
| TUT | 0.098 | 0.051 | -0.036 | -0.064 | 0.019 | 0.023 | 0.038 | -0.028 | -0.013 | 0.042 | -0.044 | 0.045 | -0.014 | -0.049 | -0.020 | -0.009 | 0.018 | 0.047 | -0.025 | 0.025 | 0.029 | 0.038 | 1.000 | 0.014 |
| ZEST | -0.006 | 0.092 | 0.003 | -0.070 | 0.004 | -0.002 | -0.022 | 0.005 | 0.025 | -0.077 | 0.045 | 0.079 | 0.014 | -0.015 | -0.053 | 0.012 | -0.039 | 0.017 | -0.004 | 0.096 | 0.024 | 0.037 | 0.014 | 1.000 |

The complete 210 × 210 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| C | CUSDT | 67.5% | 57.0% | BTCUSDT | 0.332 |
| ZORA | ZORAUSDT | 67.8% | 56.7% | BTCUSDT | 0.313 |
| QNT | QNTUSDT | 68.0% | 56.6% | BTCUSDT | 0.491 |
| GTC | GTCUSDT | 68.0% | 56.6% | BTCUSDT | 0.290 |
| 2Z | 2ZUSDT | 68.2% | 56.4% | DYDXUSDT | 0.298 |
| POLYX | POLYXUSDT | 68.3% | 56.3% | BTCUSDT | 0.479 |
| GUA | GUAUSDT | 68.3% | 56.3% | LABUSDT | 0.237 |
| XLE | XLEUSDT | 68.3% | 56.3% | JCTUSDT | -0.207 |
| SKL | SKLUSDT | 68.3% | 56.3% | MOCAUSDT | 0.172 |
| TFUEL | TFUELUSDT | 68.4% | 56.2% | BTCUSDT | 0.312 |
| ESP | ESPUSDT | 68.5% | 56.1% | PYRUSDT | -0.195 |
| CROSS | CROSSUSDT | 68.6% | 56.1% | LAUSDT | 0.242 |
| G | GUSDT | 68.6% | 56.1% | PHAUSDT | 0.274 |
| REQ | REQUSDT | 68.6% | 56.0% | BTCUSDT | 0.312 |
| APE | APEUSDT | 68.6% | 56.0% | BTCUSDT | 0.262 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

