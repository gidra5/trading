# Binance portfolio basis

Generated 2026-07-23T18:18:33.424Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-06-23T00:00Z through 2026-07-22T23:00Z
- Sampling: 1h log returns (720 samples over 30 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 640 eligible assets from 715 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.2B | 42.5% |
| 2 | BFUSD | BFUSDUSDT (spot) | spot | 100.0% | 0.002 | BTCUSDT | 100.0% | 27 | 946.5K | 0.9% |
| 3 | V | VUSDT (usdm-futures) | usdm-futures | 100.0% | 0.008 | BTCUSDT | 100.0% | 3 | 292K | 30.9% |
| 4 | ATM | ATMUSDT (spot) | spot | 100.0% | 0.021 | VUSDT | 100.0% | 2 | 1.7M | 238.2% |
| 5 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 100.0% | 0.016 | BTCUSDT | 100.0% | 1 | 8.3M | 339.3% |
| 6 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.025 | BTCUSDT | 100.0% | 1 | 14M | 303.8% |
| 7 | HEI | HEIUSDT (spot) | spot, usdm-futures | 99.9% | 0.036 | ATMUSDT | 100.0% | 1 | 4.1M | 246.4% |
| 8 | B | BUSDT (usdm-futures) | usdm-futures | 99.8% | 0.048 | DEXEUSDT | 100.0% | 1 | 5.1M | 299.5% |
| 9 | XNO | XNOUSDT (spot) | spot | 99.8% | 0.045 | DEXEUSDT | 100.0% | 5 | 42.3K | 233.9% |
| 10 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.7% | 0.064 | BTCUSDT | 100.0% | 1 | 3.6M | 378.2% |
| 11 | M | MUSDT (usdm-futures) | usdm-futures | 99.6% | 0.074 | BTCUSDT | 100.0% | 1 | 12.4M | 721.8% |
| 12 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 99.6% | 0.061 | VUSDT | 100.0% | 2 | 941.3K | 478.7% |
| 13 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 99.6% | 0.052 | BTCUSDT | 100.0% | 1 | 3.5M | 139.5% |
| 14 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 99.5% | 0.072 | BTCUSDT | 100.0% | 1 | 22.7M | 316.2% |
| 15 | KGST | KGSTUSDT (spot) | spot | 99.4% | 0.067 | MAGMAUSDT | 100.0% | 21 | 87.2K | 4.8% |
| 16 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 99.3% | 0.058 | ATMUSDT | 100.0% | 2 | 1.7M | 262.0% |
| 17 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 99.2% | 0.080 | BTCUSDT | 100.0% | 4 | 1.1M | 133.8% |
| 18 | TLM | TLMUSDT (spot) | spot, usdm-futures | 99.2% | 0.064 | BTCUSDT | 100.0% | 2 | 3.8M | 408.1% |
| 19 | BTTC | BTTCUSDT (spot) | spot | 99.1% | 0.088 | BTCUSDT | 100.0% | 21 | 173.6K | 226.0% |
| 20 | DODO | DODOUSDT (spot) | spot | 99.0% | 0.106 | BTCUSDT | 100.0% | 1 | 1.2M | 246.2% |
| 21 | PIVX | PIVXUSDT (spot) | spot | 99.0% | 0.060 | MUSDT | 100.0% | 2 | 316.8K | 280.2% |
| 22 | IN | INUSDT (usdm-futures) | usdm-futures | 98.9% | 0.092 | BTCUSDT | 100.0% | 1 | 6.5M | 300.4% |
| 23 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 98.7% | 0.095 | BUSDT | 100.0% | 1 | 24M | 690.6% |
| 24 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 98.7% | 0.100 | BTCUSDT | 100.0% | 1 | 3.4M | 95.6% |
| 25 | UB | UBUSDT (usdm-futures) | usdm-futures | 98.4% | 0.094 | BTCUSDT | 100.0% | 1 | 19.9M | 261.3% |
| 26 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 98.4% | 0.095 | BTCUSDT | 100.0% | 1 | 1.9M | 116.6% |
| 27 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 98.3% | 0.091 | BFUSDUSDT | 100.0% | 1 | 1.2M | 192.3% |
| 28 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 98.3% | 0.116 | BTCUSDT | 100.0% | 1 | 812K | 219.7% |
| 29 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 98.1% | 0.096 | STABLEUSDT | 100.0% | 2 | 18.7M | 42.5% |
| 30 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 98.0% | 0.107 | TLMUSDT | 100.0% | 2 | 3.5M | 421.1% |
| 31 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 97.9% | 0.099 | HEIUSDT | 100.0% | 1 | 3.7M | 222.0% |
| 32 | BR | BRUSDT (usdm-futures) | usdm-futures | 97.9% | 0.096 | MAGMAUSDT | 100.0% | 1 | 1.6M | 139.3% |
| 33 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 97.6% | 0.113 | BTCUSDT | 100.0% | 2 | 2.9M | 148.8% |
| 34 | DKNG | DKNGUSDT (usdm-futures) | usdm-futures | 97.5% | 0.103 | VUSDT | 100.0% | 3 | 271.2K | 56.9% |
| 35 | B2 | B2USDT (usdm-futures) | usdm-futures | 97.5% | 0.104 | BTCUSDT | 100.0% | 1 | 924.6K | 106.9% |
| 36 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 97.3% | 0.114 | BTCUSDT | 100.0% | 1 | 894.3K | 102.3% |
| 37 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 97.2% | 0.142 | BTCUSDT | 100.0% | 1 | 13.1M | 230.4% |
| 38 | CATI | CATIUSDT (spot) | spot, usdm-futures | 97.2% | 0.078 | UBUSDT | 100.0% | 2 | 385.8K | 104.3% |
| 39 | G | GUSDT (spot) | spot, usdm-futures | 97.0% | 0.153 | BTCUSDT | 100.0% | 3 | 559.4K | 178.8% |
| 40 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 97.0% | 0.145 | INUSDT | 100.0% | 1 | 2.7M | 178.6% |
| 41 | PUNDIX | PUNDIXUSDT (spot) | spot, usdm-futures | 97.0% | 0.177 | BTCUSDT | 100.0% | 2 | 228.6K | 149.0% |
| 42 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.8% | 0.125 | BTCUSDT | 100.0% | 2 | 1.4M | 171.0% |
| 43 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 96.6% | 0.138 | BTCUSDT | 100.0% | 1 | 2.9M | 173.4% |
| 44 | GPS | GPSUSDT (spot) | spot, usdm-futures | 96.5% | 0.126 | EPICUSDT | 100.0% | 2 | 431.5K | 104.3% |
| 45 | US | USUSDT (usdm-futures) | usdm-futures | 96.5% | 0.113 | UBUSDT | 100.0% | 1 | 23M | 271.5% |
| 46 | CRM | CRMUSDT (usdm-futures) | usdm-futures | 96.4% | 0.120 | VUSDT | 100.0% | 3 | 440.2K | 48.6% |
| 47 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 96.3% | 0.125 | MUSDT | 100.0% | 1 | 14.2M | 264.9% |
| 48 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 96.3% | 0.102 | PIVXUSDT | 100.0% | 1 | 40.4M | 273.4% |
| 49 | PROM | PROMUSDT (spot) | spot, usdm-futures | 96.1% | 0.156 | DEXEUSDT | 100.0% | 2 | 229.4K | 157.3% |
| 50 | GLMR | GLMRUSDT (spot) | spot | 96.1% | 0.183 | BTCUSDT | 100.0% | 12 | 318.6K | 136.8% |
| 51 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 96.0% | 0.186 | VUSDT | 100.0% | 1 | 506.9K | 24.9% |
| 52 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 95.8% | 0.122 | BTCUSDT | 100.0% | 1 | 20.1M | 428.3% |
| 53 | TA | TAUSDT (usdm-futures) | usdm-futures | 95.7% | 0.117 | AGTUSDT | 100.0% | 1 | 1.6M | 111.8% |
| 54 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 95.5% | 0.150 | TAIKOUSDT | 100.0% | 1 | 3.3M | 186.1% |
| 55 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 95.2% | 0.122 | XPINUSDT | 100.0% | 1 | 9.4M | 283.3% |
| 56 | DGB | DGBUSDT (spot) | spot | 95.2% | 0.118 | BTCUSDT | 100.0% | 5 | 80.3K | 169.9% |
| 57 | AIN | AINUSDT (usdm-futures) | usdm-futures | 95.2% | 0.157 | BTCUSDT | 100.0% | 1 | 1.3M | 167.3% |
| 58 | TAC | TACUSDT (usdm-futures) | usdm-futures | 95.1% | 0.130 | SKYAIUSDT | 100.0% | 1 | 12.8M | 854.7% |
| 59 | PORTO | PORTOUSDT (spot) | spot | 95.1% | 0.177 | BTCUSDT | 100.0% | 3 | 215.8K | 183.4% |
| 60 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 95.1% | 0.153 | BTCUSDT | 100.0% | 2 | 2.2M | 166.1% |
| 61 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 94.9% | 0.115 | BTCUSDT | 100.0% | 1 | 3.2M | 182.9% |
| 62 | BAS | BASUSDT (usdm-futures) | usdm-futures | 94.9% | 0.103 | CRWDUSDT | 100.0% | 1 | 6.6M | 327.8% |
| 63 | RPL | RPLUSDT (spot) | spot, usdm-futures | 94.6% | 0.173 | TLMUSDT | 100.0% | 9 | 272K | 174.0% |
| 64 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 94.5% | 0.195 | BTCUSDT | 100.0% | 2 | 197.4K | 33.0% |
| 65 | APR | APRUSDT (usdm-futures) | usdm-futures | 94.5% | 0.174 | BTCUSDT | 100.0% | 2 | 3M | 121.3% |
| 66 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 94.4% | 0.146 | BTCUSDT | 100.0% | 3 | 1.5M | 166.3% |
| 67 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 94.2% | 0.201 | BTCUSDT | 100.0% | 3 | 1.1M | 142.4% |
| 68 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 94.2% | 0.141 | SKYAIUSDT | 100.0% | 1 | 2M | 203.4% |
| 69 | JST | JSTUSDT (spot) | spot, usdm-futures | 94.0% | 0.165 | BTCUSDT | 100.0% | 1 | 2.7M | 43.6% |
| 70 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 93.9% | 0.153 | USUSDT | 100.0% | 1 | 3.3M | 190.1% |
| 71 | SYN | SYNUSDT (spot) | spot, usdm-futures | 93.8% | 0.138 | HEIUSDT | 100.0% | 1 | 14M | 373.7% |
| 72 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 93.7% | 0.153 | BTCUSDT | 100.0% | 1 | 725K | 188.0% |
| 73 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 93.6% | 0.203 | BTCUSDT | 100.0% | 1 | 1.7M | 56.1% |
| 74 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 93.5% | 0.253 | BTCUSDT | 100.0% | 1 | 2M | 125.0% |
| 75 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 93.3% | 0.244 | BTCUSDT | 100.0% | 2 | 234.2K | 157.7% |
| 76 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 93.2% | 0.185 | BTCUSDT | 100.0% | 1 | 1.9M | 140.6% |
| 77 | U | UUSDT (spot) | spot | 93.0% | 0.122 | AERGOUSDT | 100.0% | 12 | 15.6M | 0.7% |
| 78 | ON | ONUSDT (usdm-futures) | usdm-futures | 92.6% | 0.143 | BTCUSDT | 100.0% | 1 | 1.5M | 200.4% |
| 79 | ACT | ACTUSDT (spot) | spot, usdm-futures | 92.5% | 0.246 | BTCUSDT | 100.0% | 3 | 672.3K | 201.9% |
| 80 | HOT | HOTUSDT (spot) | spot, usdm-futures | 92.4% | 0.226 | BTCUSDT | 100.0% | 6 | 323.5K | 154.1% |
| 81 | HD | HDUSDT (usdm-futures) | usdm-futures | 92.2% | 0.210 | VUSDT | 100.0% | 1 | 279.9K | 32.8% |
| 82 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 92.1% | 0.195 | VUSDT | 100.0% | 2 | 1.1M | 47.9% |
| 83 | STAR | STARUSDT (usdm-futures) | usdm-futures | 92.1% | 0.188 | BTCUSDT | 100.0% | 0 | 1.8M | 186.5% |
| 84 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 92.0% | 0.157 | BTCUSDT | 100.0% | 1 | 51.8M | 268.4% |
| 85 | OGN | OGNUSDT (spot) | spot, usdm-futures | 91.9% | 0.266 | BTCUSDT | 100.0% | 2 | 927.9K | 160.0% |
| 86 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 91.9% | 0.174 | BTCUSDT | 100.0% | 2 | 1.2M | 99.9% |
| 87 | STRAX | STRAXUSDT (spot) | spot | 91.7% | 0.256 | BTCUSDT | 100.0% | 2 | 494.5K | 126.7% |
| 88 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 91.6% | 0.148 | AINUSDT | 100.0% | 2 | 1M | 195.7% |
| 89 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 91.5% | 0.183 | BTCUSDT | 100.0% | 2 | 1.6M | 154.2% |
| 90 | 4 | 4USDT (usdm-futures) | usdm-futures | 91.4% | 0.227 | BTCUSDT | 100.0% | 1 | 2.2M | 146.2% |
| 91 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 91.4% | 0.228 | BTCUSDT | 100.0% | 1 | 1.6M | 199.7% |
| 92 | XLE | XLEUSDT (usdm-futures) | usdm-futures | 91.4% | 0.136 | EDGEUSDT | 100.0% | 2 | 164.1K | 27.2% |
| 93 | BAR | BARUSDT (spot) | spot | 91.3% | 0.192 | ATMUSDT | 100.0% | 3 | 459.4K | 106.9% |
| 94 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 91.1% | 0.132 | ALCHUSDT | 100.0% | 2 | 14M | 248.4% |
| 95 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 91.0% | 0.167 | BTCUSDT | 100.0% | 1 | 2.3M | 148.1% |
| 96 | GNO | GNOUSDT (spot) | spot | 91.0% | 0.255 | BTCUSDT | 100.0% | 2 | 108.5K | 65.7% |
| 97 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 90.9% | 0.183 | BTCUSDT | 100.0% | 1 | 4.1M | 219.6% |
| 98 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 90.4% | 0.169 | ESPORTSUSDT | 100.0% | 2 | 2M | 118.6% |
| 99 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 90.4% | 0.202 | BTCUSDT | 100.0% | 1 | 8.1M | 229.2% |
| 100 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 90.3% | 0.192 | CRWDUSDT | 100.0% | 1 | 7.1M | 205.8% |
| 101 | BAN | BANUSDT (usdm-futures) | usdm-futures | 90.3% | 0.205 | BTCUSDT | 100.0% | 2 | 2.2M | 99.2% |
| 102 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 90.1% | 0.158 | DKNGUSDT | 100.0% | 1 | 400.4K | 35.5% |
| 103 | ACE | ACEUSDT (spot) | spot, usdm-futures | 89.9% | 0.229 | BTCUSDT | 100.0% | 2 | 318.9K | 189.4% |
| 104 | QUICK | QUICKUSDT (spot) | spot | 89.9% | 0.194 | BRUSDT | 100.0% | 2 | 103.3K | 163.4% |
| 105 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 89.8% | 0.190 | BTCUSDT | 100.0% | 2 | 1.4M | 112.6% |
| 106 | PYR | PYRUSDT (spot) | spot | 89.8% | 0.186 | XPINUSDT | 100.0% | 6 | 893.9K | 208.8% |
| 107 | THE | THEUSDT (spot) | spot, usdm-futures | 89.8% | 0.218 | BTCUSDT | 100.0% | 2 | 681.7K | 207.4% |
| 108 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 89.7% | 0.158 | EPICUSDT | 100.0% | 1 | 2.7M | 289.2% |
| 109 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 89.6% | 0.178 | BTCUSDT | 100.0% | 1 | 1.4M | 164.5% |
| 110 | Q | QUSDT (usdm-futures) | usdm-futures | 89.5% | 0.154 | BTCUSDT | 100.0% | 1 | 1.1M | 102.5% |
| 111 | BEL | BELUSDT (spot) | spot, usdm-futures | 89.4% | 0.205 | BTCUSDT | 100.0% | 1 | 1.3M | 184.9% |
| 112 | AI | AIUSDT (spot) | spot | 89.1% | 0.204 | BTCUSDT | 100.0% | 5 | 314.8K | 147.9% |
| 113 | SXT | SXTUSDT (spot) | spot, usdm-futures | 89.0% | 0.237 | BTCUSDT | 100.0% | 2 | 718.2K | 156.2% |
| 114 | SKL | SKLUSDT (spot) | spot, usdm-futures | 89.0% | 0.285 | BTCUSDT | 100.0% | 4 | 545.3K | 153.1% |
| 115 | LAB | LABUSDT (usdm-futures) | usdm-futures | 88.9% | 0.193 | TACUSDT | 100.0% | 0 | 389.8M | 521.9% |
| 116 | ID | IDUSDT (spot) | spot, usdm-futures | 88.5% | 0.192 | BTCUSDT | 100.0% | 2 | 1.1M | 137.1% |
| 117 | AWE | AWEUSDT (spot) | spot, usdm-futures | 88.5% | 0.174 | BTCUSDT | 100.0% | 1 | 390.4K | 108.3% |
| 118 | XEC | XECUSDT (spot) | spot, usdm-futures | 88.4% | 0.206 | BTCUSDT | 100.0% | 2 | 317.1K | 164.7% |
| 119 | ONG | ONGUSDT (spot) | spot, usdm-futures | 88.2% | 0.236 | BTCUSDT | 100.0% | 1 | 226.6K | 121.7% |
| 120 | MMT | MMTUSDT (spot) | spot, usdm-futures | 88.2% | 0.167 | LUMIAUSDT | 100.0% | 1 | 1.1M | 141.5% |
| 121 | ERA | ERAUSDT (spot) | spot, usdm-futures | 88.1% | 0.292 | BTCUSDT | 100.0% | 3 | 229.7K | 174.8% |
| 122 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 88.1% | 0.240 | BTCUSDT | 100.0% | 1 | 4.4M | 162.3% |
| 123 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 87.8% | 0.206 | BTCUSDT | 100.0% | 1 | 16.1M | 205.5% |
| 124 | BANK | BANKUSDT (spot) | spot, usdm-futures | 87.7% | 0.257 | DEXEUSDT | 100.0% | 3 | 477.4K | 432.1% |
| 125 | FF | FFUSDT (spot) | spot, usdm-futures | 87.4% | 0.169 | BTCUSDT | 100.0% | 1 | 1.2M | 73.1% |
| 126 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 87.4% | 0.173 | ACTUSDT | 100.0% | 1 | 4.5M | 159.2% |
| 127 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 87.2% | 0.246 | BTCUSDT | 100.0% | 2 | 1.9M | 121.4% |
| 128 | SENT | SENTUSDT (spot) | spot, usdm-futures | 87.1% | 0.251 | BTCUSDT | 100.0% | 2 | 1.4M | 116.3% |
| 129 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 86.8% | 0.249 | BTCUSDT | 100.0% | 1 | 2.5M | 155.8% |
| 130 | QKC | QKCUSDT (spot) | spot | 86.8% | 0.223 | PHAROSUSDT | 100.0% | 2 | 111.1K | 144.8% |
| 131 | AUDIO | AUDIOUSDT (spot) | spot | 86.7% | 0.283 | BTCUSDT | 100.0% | 2 | 295.9K | 105.0% |
| 132 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 86.7% | 0.154 | BTCUSDT | 100.0% | 1 | 1.9M | 199.3% |
| 133 | RIF | RIFUSDT (spot) | spot, usdm-futures | 86.7% | 0.183 | DEXEUSDT | 100.0% | 2 | 2M | 252.9% |
| 134 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 86.6% | 0.278 | BTCUSDT | 100.0% | 1 | 21.5M | 71.8% |
| 135 | IQ | IQUSDT (spot) | spot | 86.1% | 0.253 | BTCUSDT | 100.0% | 2 | 54.4K | 74.5% |
| 136 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 86.1% | 0.205 | BTCUSDT | 100.0% | 1 | 3.3M | 187.0% |
| 137 | SUN | SUNUSDT (spot) | spot, usdm-futures | 86.0% | 0.291 | BTCUSDT | 100.0% | 3 | 666.5K | 25.8% |
| 138 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 85.9% | 0.171 | TAIKOUSDT | 100.0% | 1 | 45.5M | 396.0% |
| 139 | OPG | OPGUSDT (spot) | spot, usdm-futures | 85.6% | 0.227 | BTCUSDT | 100.0% | 2 | 1.7M | 175.2% |
| 140 | NOM | NOMUSDT (spot) | spot, usdm-futures | 85.5% | 0.282 | BTCUSDT | 100.0% | 5 | 600.4K | 170.0% |
| 141 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 85.2% | 0.174 | BTCUSDT | 100.0% | 4 | 900.6K | 126.6% |
| 142 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 85.2% | 0.293 | BTCUSDT | 100.0% | 2 | 338.9K | 134.2% |
| 143 | XAN | XANUSDT (usdm-futures) | usdm-futures | 85.0% | 0.242 | BTCUSDT | 100.0% | 0 | 2.9M | 164.4% |
| 144 | AT | ATUSDT (spot) | spot, usdm-futures | 84.8% | 0.202 | BTCUSDT | 100.0% | 2 | 317.8K | 88.6% |
| 145 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 84.7% | 0.218 | BTCUSDT | 100.0% | 2 | 938.2K | 122.5% |
| 146 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 84.6% | 0.214 | BTCUSDT | 100.0% | 2 | 3.4M | 154.3% |
| 147 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 84.4% | 0.274 | BTCUSDT | 100.0% | 1 | 745.8K | 97.3% |
| 148 | KITE | KITEUSDT (spot) | spot, usdm-futures | 84.3% | 0.286 | BTCUSDT | 100.0% | 1 | 4M | 121.3% |
| 149 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 84.1% | 0.191 | UBUSDT | 100.0% | 1 | 1.5M | 214.2% |
| 150 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 84.0% | 0.194 | GPSUSDT | 100.0% | 1 | 3.5M | 217.1% |
| 151 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 83.9% | 0.245 | EDGEUSDT | 100.0% | 1 | 2.8M | 327.3% |
| 152 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 83.9% | 0.151 | BTCUSDT | 100.0% | 1 | 1.1M | 142.6% |
| 153 | CC | CCUSDT (usdm-futures) | usdm-futures | 83.7% | 0.222 | BTCUSDT | 100.0% | 1 | 4M | 61.2% |
| 154 | RE | REUSDT (spot) | spot, usdm-futures | 83.7% | 0.257 | BTCUSDT | 100.0% | 1 | 20.8M | 180.3% |
| 155 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 83.2% | 0.229 | BTCUSDT | 100.0% | 1 | 9.5M | 222.3% |
| 156 | TUT | TUTUSDT (spot) | spot, usdm-futures | 83.2% | 0.285 | BTCUSDT | 100.0% | 2 | 457.8K | 103.3% |
| 157 | REQ | REQUSDT (spot) | spot | 83.1% | 0.401 | BTCUSDT | 100.0% | 5 | 87.8K | 69.5% |
| 158 | ONE | ONEUSDT (spot) | spot, usdm-futures | 83.0% | 0.352 | BTCUSDT | 100.0% | 6 | 194.4K | 153.2% |
| 159 | ZKP | ZKPUSDT (spot) | spot, usdm-futures | 82.9% | 0.271 | BTCUSDT | 100.0% | 3 | 335.2K | 131.2% |
| 160 | GME | GMEUSDT (usdm-futures) | usdm-futures | 82.9% | 0.210 | BTCUSDT | 100.0% | 2 | 175.7K | 31.5% |
| 161 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 82.8% | 0.281 | BTCUSDT | 100.0% | 2 | 1.3M | 117.3% |
| 162 | FTT | FTTUSDT (spot) | spot | 82.7% | 0.320 | BTCUSDT | 100.0% | 2 | 200.6K | 122.2% |
| 163 | BX | BXUSDT (usdm-futures) | usdm-futures | 82.6% | 0.178 | BTCUSDT | 100.0% | 1 | 573.5K | 47.6% |
| 164 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 82.5% | 0.327 | BTCUSDT | 100.0% | 2 | 1.2M | 128.7% |
| 165 | DCR | DCRUSDT (spot) | spot | 82.4% | 0.274 | BTCUSDT | 100.0% | 2 | 198.7K | 128.0% |
| 166 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 82.3% | 0.276 | BTCUSDT | 100.0% | 2 | 475.6K | 154.2% |
| 167 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 82.2% | 0.304 | BTCUSDT | 100.0% | 0 | 1.7M | 157.7% |
| 168 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 82.1% | 0.196 | BTCUSDT | 100.0% | 0 | 13.9M | 191.7% |
| 169 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 82.0% | 0.235 | BTCUSDT | 100.0% | 1 | 499.3K | 87.7% |
| 170 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 81.8% | 0.246 | BROCCOLIF3BUSDT | 100.0% | 1 | 4.1M | 243.5% |
| 171 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 81.7% | 0.327 | BTCUSDT | 100.0% | 1 | 1.2M | 120.1% |
| 172 | T | TUSDT (spot) | spot, usdm-futures | 81.4% | 0.289 | BTCUSDT | 100.0% | 4 | 468.8K | 153.7% |
| 173 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 81.1% | 0.354 | BTCUSDT | 100.0% | 1 | 776.7K | 83.4% |
| 174 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 81.1% | 0.217 | BTCUSDT | 100.0% | 1 | 2M | 96.6% |
| 175 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 80.7% | 0.371 | BTCUSDT | 100.0% | 4 | 618.1K | 106.2% |
| 176 | POWR | POWRUSDT (spot) | spot, usdm-futures | 80.6% | 0.326 | BTCUSDT | 100.0% | 3 | 157.5K | 130.4% |
| 177 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 80.2% | 0.271 | BTCUSDT | 100.0% | 2 | 2.4M | 85.8% |
| 178 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 80.2% | 0.301 | BTCUSDT | 100.0% | 1 | 604.9K | 68.5% |
| 179 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 80.1% | 0.238 | CRWDUSDT | 100.0% | 3 | 379.2K | 26.6% |
| 180 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 80.0% | 0.291 | BTCUSDT | 100.0% | 2 | 1.3M | 136.6% |
| 181 | AMP | AMPUSDT (spot) | spot | 79.9% | 0.256 | BTCUSDT | 100.0% | 5 | 382.5K | 95.4% |
| 182 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 79.8% | 0.332 | BTCUSDT | 100.0% | 1 | 1.7M | 137.8% |
| 183 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 79.6% | 0.273 | BTCUSDT | 100.0% | 1 | 3M | 187.2% |
| 184 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 79.3% | 0.290 | BTCUSDT | 100.0% | 5 | 722.4K | 153.3% |
| 185 | JTO | JTOUSDT (spot) | spot, usdm-futures | 79.2% | 0.285 | BTCUSDT | 100.0% | 1 | 5.4M | 129.7% |
| 186 | GUN | GUNUSDT (spot) | spot, usdm-futures | 79.2% | 0.380 | BTCUSDT | 100.0% | 3 | 678.4K | 125.7% |
| 187 | QI | QIUSDT (spot) | spot | 79.0% | 0.386 | PIVXUSDT | 100.0% | 10 | 140.9K | 136.0% |
| 188 | MBL | MBLUSDT (spot) | spot | 78.9% | 0.255 | QKCUSDT | 100.0% | 3 | 660.2K | 97.9% |
| 189 | STG | STGUSDT (spot) | spot, usdm-futures | 78.8% | 0.278 | BTCUSDT | 100.0% | 2 | 733.6K | 120.7% |
| 190 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 78.5% | 0.201 | GMEUSDT | 100.0% | 4 | 222.8K | 33.6% |
| 191 | LIT | LITUSDT (usdm-futures) | usdm-futures | 78.3% | 0.378 | BTCUSDT | 100.0% | 1 | 49.7M | 138.7% |
| 192 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 78.3% | 0.440 | BTCUSDT | 100.0% | 2 | 948.8K | 123.2% |
| 193 | H | HUSDT (usdm-futures) | usdm-futures | 78.1% | 0.323 | MUSDT | 100.0% | 1 | 12.2M | 281.4% |
| 194 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 77.9% | 0.380 | BTCUSDT | 100.0% | 1 | 1.2M | 128.8% |
| 195 | NMR | NMRUSDT (spot) | spot, usdm-futures | 77.7% | 0.372 | BTCUSDT | 100.0% | 3 | 303.5K | 74.7% |
| 196 | COPPER | COPPERUSDT (usdm-futures) | usdm-futures | 77.6% | 0.382 | BTCUSDT | 100.0% | 2 | 4.9M | 23.8% |
| 197 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 77.4% | 0.365 | BTCUSDT | 100.0% | 3 | 2M | 97.1% |
| 198 | HYPER | HYPERUSDT (spot) | spot, usdm-futures | 77.1% | 0.390 | BTCUSDT | 100.0% | 3 | 1.1M | 88.4% |
| 199 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 76.9% | 0.323 | BTCUSDT | 100.0% | 2 | 664.1K | 104.3% |
| 200 | TREE | TREEUSDT (spot) | spot, usdm-futures | 76.7% | 0.366 | BTCUSDT | 100.0% | 3 | 604.3K | 129.1% |
| 201 | CITY | CITYUSDT (spot) | spot | 76.6% | 0.295 | BARUSDT | 100.0% | 3 | 755.5K | 119.8% |
| 202 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 76.5% | 0.229 | IDOLUSDT | 100.0% | 1 | 721.8K | 79.1% |
| 203 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 76.1% | 0.248 | CRWDUSDT | 100.0% | 3 | 8.3M | 315.8% |
| 204 | XPL | XPLUSDT (spot) | spot, usdm-futures | 75.9% | 0.424 | BTCUSDT | 100.0% | 1 | 9.8M | 125.7% |
| 205 | MITO | MITOUSDT (spot) | spot, usdm-futures | 75.7% | 0.290 | BTCUSDT | 100.0% | 2 | 784.5K | 121.6% |
| 206 | SKY | SKYUSDT (spot) | spot, usdm-futures | 75.6% | 0.459 | BTCUSDT | 100.0% | 1 | 1.2M | 74.8% |
| 207 | HEMI | HEMIUSDT (spot) | spot, usdm-futures | 75.4% | 0.282 | TOWNSUSDT | 100.0% | 2 | 789.1K | 159.9% |
| 208 | TST | TSTUSDT (spot) | spot, usdm-futures | 75.1% | 0.307 | BTCUSDT | 100.0% | 2 | 595.4K | 111.4% |
| 209 | CELO | CELOUSDT (spot) | spot, usdm-futures | 75.1% | 0.350 | BTCUSDT | 100.0% | 2 | 951.7K | 115.4% |
| 210 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 74.8% | 0.375 | BTCUSDT | 100.0% | 2 | 3M | 127.1% |
| 211 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 74.6% | 0.369 | BTCUSDT | 100.0% | 2 | 674.3K | 101.2% |
| 212 | WIN | WINUSDT (spot) | spot | 74.3% | 0.468 | BTCUSDT | 100.0% | 3 | 98.2K | 54.4% |
| 213 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 74.2% | 0.445 | BTCUSDT | 100.0% | 2 | 522.8K | 99.3% |
| 214 | POPCAT | POPCATUSDT (usdm-futures) | usdm-futures | 74.1% | 0.450 | BTCUSDT | 100.0% | 1 | 2.1M | 121.1% |
| 215 | GTC | GTCUSDT (spot) | spot, usdm-futures | 74.1% | 0.367 | BTCUSDT | 100.0% | 9 | 126.2K | 122.8% |
| 216 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 73.8% | 0.357 | BTCUSDT | 100.0% | 2 | 705K | 220.6% |
| 217 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 73.7% | 0.349 | BTCUSDT | 100.0% | 1 | 1.7M | 106.6% |
| 218 | GNS | GNSUSDT (spot) | spot | 73.6% | 0.410 | BTCUSDT | 100.0% | 5 | 64.4K | 49.3% |
| 219 | MET | METUSDT (spot) | spot, usdm-futures | 73.5% | 0.382 | BTCUSDT | 100.0% | 1 | 1.1M | 134.2% |
| 220 | OPN | OPNUSDT (spot) | spot, usdm-futures | 73.3% | 0.314 | BTCUSDT | 100.0% | 3 | 4.8M | 128.8% |
| 221 | ARK | ARKUSDT (spot) | spot, usdm-futures | 73.2% | 0.401 | BTCUSDT | 100.0% | 2 | 56.9K | 116.9% |
| 222 | ELSA | ELSAUSDT (usdm-futures) | usdm-futures | 73.1% | 0.324 | BTCUSDT | 100.0% | 1 | 2.7M | 111.6% |
| 223 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 73.0% | 0.272 | BTCUSDT | 100.0% | 6 | 468.4K | 75.1% |
| 224 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 72.9% | 0.375 | BTCUSDT | 100.0% | 1 | 1.2M | 110.8% |
| 225 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 72.8% | 0.293 | BXUSDT | 100.0% | 2 | 383.7K | 28.8% |
| 226 | BOB | 1000000BOBUSDT (usdm-futures) | usdm-futures | 72.7% | 0.380 | BTCUSDT | 100.0% | 2 | 632.1K | 88.1% |
| 227 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 72.5% | 0.399 | BTCUSDT | 100.0% | 1 | 3.7M | 109.6% |
| 228 | KAT | KATUSDT (spot) | spot, usdm-futures | 72.5% | 0.383 | BTCUSDT | 100.0% | 2 | 900.3K | 102.7% |
| 229 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 72.2% | 0.341 | BTCUSDT | 100.0% | 2 | 14.3M | 218.2% |
| 230 | GRASS | GRASSUSDT (usdm-futures) | usdm-futures | 72.2% | 0.384 | BTCUSDT | 100.0% | 2 | 10.3M | 155.2% |
| 231 | C | CUSDT (spot) | spot, usdm-futures | 72.1% | 0.384 | BTCUSDT | 100.0% | 3 | 332K | 86.6% |
| 232 | BZ | BZUSDT (usdm-futures) | usdm-futures | 71.9% | 0.406 | XLEUSDT | 100.0% | 2 | 119.2M | 44.6% |
| 233 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 71.8% | 0.384 | BTCUSDT | 100.0% | 1 | 682K | 91.8% |
| 234 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 71.7% | 0.395 | BTCUSDT | 100.0% | 1 | 1.1M | 98.8% |
| 235 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 71.6% | 0.314 | BTCUSDT | 100.0% | 2 | 7.4M | 195.1% |
| 236 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 71.4% | 0.469 | BTCUSDT | 100.0% | 5 | 622K | 70.1% |
| 237 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 71.3% | 0.204 | CRMUSDT | 100.0% | 2 | 336.3K | 41.4% |
| 238 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 71.2% | 0.274 | BTCUSDT | 100.0% | 1 | 12.2M | 160.2% |
| 239 | APE | APEUSDT (spot) | spot, usdm-futures | 71.0% | 0.401 | BTCUSDT | 100.0% | 2 | 1.1M | 94.2% |
| 240 | HFT | HFTUSDT (spot) | spot, usdm-futures | 70.9% | 0.317 | BTCUSDT | 100.0% | 5 | 180.5K | 138.2% |
| 241 | RED | REDUSDT (spot) | spot, usdm-futures | 70.4% | 0.448 | BTCUSDT | 100.0% | 3 | 495.2K | 99.8% |
| 242 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 70.4% | 0.234 | BTCUSDT | 100.0% | 3 | 435.2K | 207.3% |
| 243 | BBX | BBXUSDT (usdm-futures) | usdm-futures | 70.2% | 0.248 | BXUSDT | 100.0% | 2 | 2.7M | 116.9% |
| 244 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 70.1% | 0.344 | BTCUSDT | 100.0% | 1 | 4.5M | 147.1% |
| 245 | RARE | RAREUSDT (spot) | spot, usdm-futures | 69.9% | 0.390 | BTCUSDT | 100.0% | 12 | 201.4K | 95.6% |
| 246 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 69.9% | 0.447 | BTCUSDT | 100.0% | 3 | 3.6M | 54.1% |
| 247 | VANA | VANAUSDT (spot) | spot, usdm-futures | 69.8% | 0.434 | BTCUSDT | 100.0% | 3 | 638.4K | 80.3% |
| 248 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 69.7% | 0.347 | HFTUSDT | 100.0% | 9 | 283.1K | 138.0% |
| 249 | LA | LAUSDT (spot) | spot, usdm-futures | 69.5% | 0.403 | ERAUSDT | 100.0% | 2 | 639.6K | 123.0% |
| 250 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 69.3% | 0.315 | BTCUSDT | 100.0% | 2 | 876.5K | 115.0% |
| 251 | ADX | ADXUSDT (spot) | spot | 69.1% | 0.425 | BTCUSDT | 100.0% | 3 | 233.2K | 63.1% |
| 252 | BREV | BREVUSDT (spot) | spot, usdm-futures | 68.8% | 0.285 | BTCUSDT | 100.0% | 3 | 311.7K | 139.5% |
| 253 | ESP | ESPUSDT (spot) | spot, usdm-futures | 68.8% | 0.400 | INUSDT | 100.0% | 1 | 523K | 93.0% |
| 254 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 68.4% | 0.402 | BTCUSDT | 100.0% | 2 | 265.2K | 80.2% |
| 255 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 68.3% | 0.445 | BTCUSDT | 100.0% | 2 | 919.2K | 98.0% |
| 256 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 68.3% | 0.391 | BTCUSDT | 100.0% | 1 | 2.9M | 117.0% |
| 257 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 68.2% | 0.347 | BTCUSDT | 100.0% | 1 | 2.5M | 132.5% |
| 258 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 68.1% | 0.305 | COPPERUSDT | 100.0% | 1 | 1.5M | 80.0% |
| 259 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 68.1% | 0.359 | DKNGUSDT | 100.0% | 2 | 330.4K | 39.5% |
| 260 | CHILLGUY | CHILLGUYUSDT (usdm-futures) | usdm-futures | 68.0% | 0.454 | BTCUSDT | 100.0% | 1 | 903.6K | 115.6% |
| 261 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 67.9% | 0.461 | BTCUSDT | 100.0% | 2 | 330.2K | 86.3% |
| 262 | HOLO | HOLOUSDT (spot) | spot, usdm-futures | 67.6% | 0.374 | BTCUSDT | 100.0% | 2 | 804.8K | 93.9% |
| 263 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 67.5% | 0.266 | CRWDUSDT | 100.0% | 1 | 17M | 33.9% |
| 264 | MINA | MINAUSDT (spot) | spot, usdm-futures | 67.3% | 0.478 | BTCUSDT | 100.0% | 2 | 392.1K | 92.5% |
| 265 | BICO | BICOUSDT (spot) | spot, usdm-futures | 67.3% | 0.328 | BTCUSDT | 100.0% | 1 | 915.9K | 151.3% |
| 266 | SC | SCUSDT (spot) | spot | 66.9% | 0.456 | BTCUSDT | 100.0% | 5 | 169.8K | 63.7% |
| 267 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 66.8% | 0.483 | BTCUSDT | 100.0% | 4 | 396.5K | 89.6% |
| 268 | FLOCK | FLOCKUSDT (usdm-futures) | usdm-futures | 66.6% | 0.459 | BTCUSDT | 100.0% | 1 | 1.3M | 117.2% |
| 269 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 66.6% | 0.450 | BTCUSDT | 100.0% | 1 | 1.6M | 114.1% |
| 270 | YB | YBUSDT (spot) | spot, usdm-futures | 66.4% | 0.395 | BTCUSDT | 100.0% | 2 | 295.3K | 109.9% |
| 271 | WET | WETUSDT (usdm-futures) | usdm-futures | 66.3% | 0.361 | BTCUSDT | 100.0% | 1 | 1.8M | 107.9% |
| 272 | DIS | DISUSDT (usdm-futures) | usdm-futures | 66.1% | 0.376 | HDUSDT | 100.0% | 2 | 224.5K | 27.5% |
| 273 | MEME | MEMEUSDT (spot) | spot, usdm-futures | 66.1% | 0.419 | BTCUSDT | 100.0% | 3 | 567.4K | 84.8% |
| 274 | HPE | HPEUSDT (usdm-futures) | usdm-futures | 66.1% | 0.326 | SPCXBUSDT | 100.0% | 1 | 735.8K | 70.4% |
| 275 | XTZ | XTZUSDT (spot) | spot, usdm-futures | 66.0% | 0.539 | BTCUSDT | 100.0% | 1 | 405.7K | 78.7% |
| 276 | A | AUSDT (spot) | spot, usdm-futures | 65.7% | 0.472 | BTCUSDT | 100.0% | 2 | 440.6K | 82.9% |
| 277 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 65.4% | 0.450 | BTCUSDT | 100.0% | 1 | 119K | 67.8% |
| 278 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 65.2% | 0.493 | BTCUSDT | 100.0% | 5 | 189.5K | 82.2% |
| 279 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 65.1% | 0.402 | BTCUSDT | 100.0% | 2 | 1.8M | 75.2% |
| 280 | RAD | RADUSDT (spot) | spot | 65.0% | 0.391 | BTCUSDT | 100.0% | 7 | 216.5K | 55.1% |
| 281 | ALICE | ALICEUSDT (spot) | spot, usdm-futures | 64.9% | 0.437 | ACEUSDT | 100.0% | 2 | 1.1M | 129.5% |

## Diagnostics

- Basis size selected: 281
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.081
- Maximum pairwise absolute correlation: 0.539
- Mean whole-market projection R²: 83.5%
- Median whole-market projection R²: 80.1%
- 10th-percentile whole-market projection R²: 62.4%
- Minimum whole-market projection R²: 58.3%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.6% | 18.1% | 1.3% | 0.0% |
| 5 | 23.9% | 19.8% | 2.6% | 0.2% |
| 10 | 25.1% | 20.9% | 3.2% | 0.7% |
| 15 | 26.7% | 22.2% | 4.3% | 1.3% |
| 20 | 28.4% | 23.2% | 5.4% | 2.0% |
| 25 | 30.0% | 24.9% | 6.7% | 3.2% |
| 30 | 31.6% | 26.1% | 7.6% | 4.2% |
| 35 | 33.0% | 27.5% | 8.6% | 5.3% |
| 40 | 34.4% | 29.1% | 9.7% | 5.9% |
| 45 | 35.9% | 30.6% | 11.0% | 7.1% |
| 50 | 37.5% | 32.0% | 12.2% | 7.8% |
| 55 | 38.8% | 33.0% | 13.2% | 9.3% |
| 60 | 40.3% | 35.1% | 14.3% | 10.0% |
| 65 | 41.7% | 36.1% | 15.6% | 10.9% |
| 70 | 43.0% | 37.1% | 16.6% | 12.1% |
| 75 | 44.5% | 38.7% | 17.6% | 13.1% |
| 80 | 45.7% | 39.9% | 18.8% | 14.9% |
| 85 | 47.0% | 40.7% | 19.9% | 15.6% |
| 90 | 48.3% | 42.3% | 20.9% | 16.5% |
| 95 | 49.5% | 43.3% | 22.2% | 17.3% |
| 100 | 50.7% | 44.0% | 23.5% | 18.5% |
| 105 | 52.0% | 45.5% | 24.5% | 19.4% |
| 110 | 53.3% | 47.0% | 25.8% | 20.1% |
| 115 | 54.4% | 48.0% | 27.2% | 21.6% |
| 120 | 55.7% | 49.6% | 28.3% | 22.3% |
| 125 | 56.8% | 50.5% | 29.6% | 23.7% |
| 130 | 57.9% | 51.8% | 30.6% | 24.8% |
| 135 | 59.1% | 52.8% | 31.9% | 25.9% |
| 140 | 60.2% | 53.9% | 33.0% | 27.4% |
| 145 | 61.2% | 54.9% | 33.8% | 28.5% |
| 150 | 62.2% | 56.1% | 34.9% | 29.5% |
| 155 | 63.1% | 56.9% | 36.2% | 30.9% |
| 160 | 64.2% | 58.3% | 38.1% | 31.4% |
| 165 | 65.1% | 59.0% | 39.3% | 32.3% |
| 170 | 66.1% | 59.8% | 40.4% | 33.2% |
| 175 | 67.1% | 60.7% | 41.2% | 35.1% |
| 180 | 68.1% | 61.9% | 42.6% | 36.1% |
| 185 | 69.0% | 62.6% | 43.4% | 37.3% |
| 190 | 70.0% | 64.1% | 44.2% | 38.6% |
| 195 | 70.9% | 65.1% | 45.2% | 39.8% |
| 200 | 72.0% | 66.3% | 46.6% | 41.3% |
| 205 | 72.8% | 67.3% | 47.6% | 42.9% |
| 210 | 73.6% | 68.2% | 48.6% | 44.4% |
| 215 | 74.4% | 69.0% | 49.7% | 45.6% |
| 220 | 75.1% | 69.8% | 50.7% | 46.4% |
| 225 | 75.9% | 70.7% | 52.2% | 47.1% |
| 230 | 76.6% | 71.6% | 53.2% | 48.1% |
| 235 | 77.4% | 72.2% | 54.4% | 49.0% |
| 240 | 78.1% | 73.0% | 55.2% | 50.4% |
| 245 | 78.9% | 74.1% | 56.5% | 51.1% |
| 250 | 79.6% | 74.8% | 57.1% | 52.3% |
| 255 | 80.2% | 75.5% | 58.0% | 53.3% |
| 260 | 80.9% | 76.5% | 59.0% | 53.9% |
| 265 | 81.5% | 77.6% | 59.7% | 55.2% |
| 270 | 82.1% | 78.4% | 60.8% | 56.1% |
| 275 | 82.8% | 79.3% | 61.7% | 56.9% |
| 280 | 83.4% | 79.9% | 62.2% | 57.9% |
| 281 | 83.5% | 80.1% | 62.4% | 58.3% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | BFUSD | V | ATM | DEXE | MAGMA | HEI | B | XNO | AKE | M | CRWD | ZEREBRO | SKYAI | KGST | EPIC | HANA | TLM | BTTC | DODO | PIVX | IN | EVAA | STABLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.002 | 0.008 | 0.005 | -0.016 | 0.025 | 0.006 | 0.022 | 0.040 | 0.064 | 0.074 | -0.036 | 0.052 | 0.072 | -0.035 | 0.041 | 0.080 | 0.064 | 0.088 | 0.106 | 0.056 | 0.092 | 0.046 | 0.100 |
| BFUSD | 0.002 | 1.000 | -0.007 | -0.014 | -0.016 | -0.020 | 0.019 | -0.007 | -0.005 | -0.019 | 0.019 | -0.005 | -0.021 | 0.007 | 0.000 | 0.024 | -0.003 | -0.007 | -0.011 | 0.014 | 0.011 | -0.003 | -0.012 | -0.037 |
| V | 0.008 | -0.007 | 1.000 | 0.021 | 0.014 | 0.024 | 0.021 | 0.015 | -0.002 | 0.022 | -0.011 | -0.061 | 0.024 | -0.017 | -0.009 | 0.004 | 0.035 | 0.031 | 0.005 | 0.017 | -0.027 | 0.036 | 0.050 | 0.046 |
| ATM | 0.005 | -0.014 | 0.021 | 1.000 | -0.010 | -0.021 | -0.036 | -0.027 | -0.004 | 0.011 | 0.021 | 0.007 | 0.023 | 0.032 | -0.015 | -0.058 | 0.001 | -0.002 | 0.028 | -0.026 | 0.030 | 0.025 | 0.030 | -0.010 |
| DEXE | -0.016 | -0.016 | 0.014 | -0.010 | 1.000 | 0.010 | 0.023 | 0.048 | 0.045 | -0.020 | -0.013 | 0.005 | -0.002 | 0.008 | 0.065 | 0.020 | -0.006 | 0.013 | -0.040 | -0.011 | 0.021 | 0.003 | 0.017 | -0.003 |
| MAGMA | 0.025 | -0.020 | 0.024 | -0.021 | 0.010 | 1.000 | 0.018 | 0.001 | -0.023 | -0.011 | -0.024 | -0.042 | -0.011 | 0.024 | -0.067 | 0.015 | 0.020 | 0.036 | -0.000 | 0.019 | -0.053 | -0.028 | -0.032 | -0.008 |
| HEI | 0.006 | 0.019 | 0.021 | -0.036 | 0.023 | 0.018 | 1.000 | -0.019 | 0.002 | -0.017 | 0.003 | 0.018 | -0.042 | 0.000 | 0.024 | 0.022 | -0.022 | 0.044 | 0.021 | 0.008 | 0.043 | -0.023 | 0.028 | 0.013 |
| B | 0.022 | -0.007 | 0.015 | -0.027 | 0.048 | 0.001 | -0.019 | 1.000 | 0.028 | 0.027 | 0.013 | -0.022 | 0.036 | 0.029 | 0.013 | -0.048 | 0.011 | -0.001 | 0.020 | 0.001 | 0.005 | 0.016 | 0.095 | 0.000 |
| XNO | 0.040 | -0.005 | -0.002 | -0.004 | 0.045 | -0.023 | 0.002 | 0.028 | 1.000 | -0.012 | 0.014 | -0.001 | 0.036 | -0.008 | -0.002 | -0.018 | 0.048 | 0.005 | -0.006 | 0.020 | 0.005 | 0.023 | 0.030 | 0.025 |
| AKE | 0.064 | -0.019 | 0.022 | 0.011 | -0.020 | -0.011 | -0.017 | 0.027 | -0.012 | 1.000 | 0.010 | -0.005 | 0.002 | -0.004 | -0.010 | 0.022 | 0.057 | -0.003 | 0.004 | -0.009 | 0.012 | 0.016 | 0.012 | -0.029 |
| M | 0.074 | 0.019 | -0.011 | 0.021 | -0.013 | -0.024 | 0.003 | 0.013 | 0.014 | 0.010 | 1.000 | -0.026 | 0.004 | 0.029 | 0.008 | -0.006 | 0.002 | 0.045 | -0.005 | 0.053 | 0.060 | -0.005 | -0.005 | 0.035 |
| CRWD | -0.036 | -0.005 | -0.061 | 0.007 | 0.005 | -0.042 | 0.018 | -0.022 | -0.001 | -0.005 | -0.026 | 1.000 | 0.004 | 0.038 | -0.016 | 0.015 | -0.003 | -0.034 | 0.053 | 0.004 | 0.010 | -0.027 | -0.034 | -0.035 |
| ZEREBRO | 0.052 | -0.021 | 0.024 | 0.023 | -0.002 | -0.011 | -0.042 | 0.036 | 0.036 | 0.002 | 0.004 | 0.004 | 1.000 | -0.012 | -0.022 | -0.039 | 0.025 | 0.011 | -0.010 | 0.009 | -0.046 | -0.064 | 0.051 | -0.028 |
| SKYAI | 0.072 | 0.007 | -0.017 | 0.032 | 0.008 | 0.024 | 0.000 | 0.029 | -0.008 | -0.004 | 0.029 | 0.038 | -0.012 | 1.000 | 0.018 | -0.037 | -0.013 | 0.051 | 0.040 | 0.038 | 0.055 | 0.023 | 0.051 | 0.063 |
| KGST | -0.035 | 0.000 | -0.009 | -0.015 | 0.065 | -0.067 | 0.024 | 0.013 | -0.002 | -0.010 | 0.008 | -0.016 | -0.022 | 0.018 | 1.000 | -0.004 | 0.009 | 0.012 | 0.012 | -0.008 | -0.007 | -0.001 | 0.019 | 0.009 |
| EPIC | 0.041 | 0.024 | 0.004 | -0.058 | 0.020 | 0.015 | 0.022 | -0.048 | -0.018 | 0.022 | -0.006 | 0.015 | -0.039 | -0.037 | -0.004 | 1.000 | -0.029 | 0.053 | -0.030 | 0.053 | -0.033 | 0.005 | -0.021 | 0.002 |
| HANA | 0.080 | -0.003 | 0.035 | 0.001 | -0.006 | 0.020 | -0.022 | 0.011 | 0.048 | 0.057 | 0.002 | -0.003 | 0.025 | -0.013 | 0.009 | -0.029 | 1.000 | 0.029 | 0.003 | 0.003 | -0.004 | 0.027 | 0.019 | 0.015 |
| TLM | 0.064 | -0.007 | 0.031 | -0.002 | 0.013 | 0.036 | 0.044 | -0.001 | 0.005 | -0.003 | 0.045 | -0.034 | 0.011 | 0.051 | 0.012 | 0.053 | 0.029 | 1.000 | -0.028 | 0.028 | -0.007 | 0.024 | -0.000 | 0.001 |
| BTTC | 0.088 | -0.011 | 0.005 | 0.028 | -0.040 | -0.000 | 0.021 | 0.020 | -0.006 | 0.004 | -0.005 | 0.053 | -0.010 | 0.040 | 0.012 | -0.030 | 0.003 | -0.028 | 1.000 | 0.008 | -0.013 | -0.042 | -0.013 | 0.005 |
| DODO | 0.106 | 0.014 | 0.017 | -0.026 | -0.011 | 0.019 | 0.008 | 0.001 | 0.020 | -0.009 | 0.053 | 0.004 | 0.009 | 0.038 | -0.008 | 0.053 | 0.003 | 0.028 | 0.008 | 1.000 | 0.018 | 0.030 | 0.002 | 0.052 |
| PIVX | 0.056 | 0.011 | -0.027 | 0.030 | 0.021 | -0.053 | 0.043 | 0.005 | 0.005 | 0.012 | 0.060 | 0.010 | -0.046 | 0.055 | -0.007 | -0.033 | -0.004 | -0.007 | -0.013 | 0.018 | 1.000 | 0.022 | 0.007 | -0.033 |
| IN | 0.092 | -0.003 | 0.036 | 0.025 | 0.003 | -0.028 | -0.023 | 0.016 | 0.023 | 0.016 | -0.005 | -0.027 | -0.064 | 0.023 | -0.001 | 0.005 | 0.027 | 0.024 | -0.042 | 0.030 | 0.022 | 1.000 | 0.045 | 0.035 |
| EVAA | 0.046 | -0.012 | 0.050 | 0.030 | 0.017 | -0.032 | 0.028 | 0.095 | 0.030 | 0.012 | -0.005 | -0.034 | 0.051 | 0.051 | 0.019 | -0.021 | 0.019 | -0.000 | -0.013 | 0.002 | 0.007 | 0.045 | 1.000 | 0.013 |
| STABLE | 0.100 | -0.037 | 0.046 | -0.010 | -0.003 | -0.008 | 0.013 | 0.000 | 0.025 | -0.029 | 0.035 | -0.035 | -0.028 | 0.063 | 0.009 | 0.002 | 0.015 | 0.001 | 0.005 | 0.052 | -0.033 | 0.035 | 0.013 | 1.000 |

The complete 281 × 281 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| FLUID | FLUIDUSDT | 58.3% | 64.6% | BTCUSDT | 0.493 |
| 2Z | 2ZUSDT | 58.4% | 64.5% | BTCUSDT | 0.483 |
| SNX | SNXUSDT | 58.6% | 64.4% | BTCUSDT | 0.518 |
| PYTH | PYTHUSDT | 58.6% | 64.3% | BTCUSDT | 0.455 |
| KAS | KASUSDT | 58.7% | 64.2% | BTCUSDT | 0.512 |
| SPORTFUN | SPORTFUNUSDT | 58.9% | 64.1% | BTCUSDT | 0.354 |
| XMR | XMRUSDT | 58.9% | 64.1% | BTCUSDT | 0.436 |
| USTC | USTCUSDT | 59.0% | 64.0% | BTCUSDT | 0.462 |
| PUMP | PUMPUSDT | 59.1% | 64.0% | BTCUSDT | 0.525 |
| AMZN | AMZNUSDT | 59.1% | 64.0% | AAPLUSDT | 0.343 |
| INX | INXUSDT | 59.1% | 64.0% | BTCUSDT | 0.419 |
| SPACE | SPACEUSDT | 59.2% | 63.9% | BTCUSDT | 0.413 |
| VELODROME | VELODROMEUSDT | 59.3% | 63.8% | BTCUSDT | 0.507 |
| FLNC | FLNCUSDT | 59.3% | 63.8% | BTCUSDT | 0.356 |
| AIA | AIAUSDT | 59.4% | 63.7% | BTCUSDT | 0.496 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

