# Binance portfolio basis

Generated 2026-07-23T18:19:19.290Z.

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
- Return universe: 639 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.2B | 42.5% |
| 2 | ATM | ATMUSDT (spot) | spot | 100.0% | 0.005 | BTCUSDT | 100.0% | 2 | 1.7M | 238.2% |
| 3 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 100.0% | 0.015 | BTCUSDT | 100.0% | 2 | 3.5M | 421.1% |
| 4 | HD | HDUSDT (usdm-futures) | usdm-futures | 100.0% | 0.015 | BTCUSDT | 100.0% | 1 | 279.9K | 32.8% |
| 5 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 100.0% | 0.016 | BTCUSDT | 100.0% | 1 | 8.3M | 339.3% |
| 6 | US | USUSDT (usdm-futures) | usdm-futures | 99.9% | 0.025 | DEXEUSDT | 100.0% | 1 | 23M | 271.5% |
| 7 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.028 | USUSDT | 100.0% | 1 | 14M | 303.8% |
| 8 | HEI | HEIUSDT (spot) | spot, usdm-futures | 99.8% | 0.036 | ATMUSDT | 100.0% | 1 | 4.1M | 246.4% |
| 9 | XNO | XNOUSDT (spot) | spot | 99.7% | 0.045 | DEXEUSDT | 100.0% | 5 | 42.3K | 233.9% |
| 10 | ADBE | ADBEUSDT (usdm-futures) | usdm-futures | 99.6% | 0.042 | TAIKOUSDT | 100.0% | 1 | 762.4K | 48.1% |
| 11 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 99.6% | 0.046 | BTCUSDT | 100.0% | 1 | 24M | 690.6% |
| 12 | U | UUSDT (spot) | spot | 99.5% | 0.055 | TAIKOUSDT | 100.0% | 12 | 15.6M | 0.7% |
| 13 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.5% | 0.064 | BTCUSDT | 100.0% | 1 | 3.6M | 378.2% |
| 14 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 99.4% | 0.052 | BTCUSDT | 100.0% | 1 | 3.5M | 139.5% |
| 15 | KGST | KGSTUSDT (spot) | spot | 99.4% | 0.067 | MAGMAUSDT | 100.0% | 21 | 87.2K | 4.8% |
| 16 | BTTC | BTTCUSDT (spot) | spot | 99.3% | 0.088 | BTCUSDT | 100.0% | 21 | 173.6K | 226.0% |
| 17 | PIVX | PIVXUSDT (spot) | spot | 99.3% | 0.056 | BTCUSDT | 100.0% | 2 | 316.8K | 280.2% |
| 18 | DODO | DODOUSDT (spot) | spot | 99.2% | 0.106 | BTCUSDT | 100.0% | 1 | 1.2M | 246.2% |
| 19 | BAS | BASUSDT (usdm-futures) | usdm-futures | 99.1% | 0.066 | ATMUSDT | 100.0% | 1 | 6.6M | 327.8% |
| 20 | B | BUSDT (usdm-futures) | usdm-futures | 99.0% | 0.095 | EVAAUSDT | 100.0% | 1 | 5.1M | 299.5% |
| 21 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 98.9% | 0.080 | BTCUSDT | 100.0% | 4 | 1.1M | 133.8% |
| 22 | M | MUSDT (usdm-futures) | usdm-futures | 98.8% | 0.074 | BTCUSDT | 100.0% | 1 | 12.4M | 721.8% |
| 23 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 98.8% | 0.116 | BTCUSDT | 100.0% | 1 | 812K | 219.7% |
| 24 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 98.7% | 0.090 | BTCUSDT | 100.0% | 1 | 1.2M | 192.3% |
| 25 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 98.6% | 0.058 | ATMUSDT | 100.0% | 2 | 1.7M | 262.0% |
| 26 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 98.5% | 0.142 | BTCUSDT | 100.0% | 1 | 13.1M | 230.4% |
| 27 | CATI | CATIUSDT (spot) | spot, usdm-futures | 98.5% | 0.069 | MAGMAUSDT | 100.0% | 2 | 385.8K | 104.3% |
| 28 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 98.4% | 0.099 | HEIUSDT | 100.0% | 1 | 3.7M | 222.0% |
| 29 | TLM | TLMUSDT (spot) | spot, usdm-futures | 98.4% | 0.107 | TAIKOUSDT | 100.0% | 2 | 3.8M | 408.1% |
| 30 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 98.3% | 0.077 | HANAUSDT | 100.0% | 1 | 2.7M | 178.6% |
| 31 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 98.2% | 0.100 | BTCUSDT | 100.0% | 1 | 3.4M | 95.6% |
| 32 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 97.9% | 0.134 | BTCUSDT | 100.0% | 1 | 7.1M | 205.8% |
| 33 | B2 | B2USDT (usdm-futures) | usdm-futures | 97.8% | 0.111 | UUSDT | 100.0% | 1 | 924.6K | 106.9% |
| 34 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 97.7% | 0.114 | BTCUSDT | 100.0% | 1 | 894.3K | 102.3% |
| 35 | BR | BRUSDT (usdm-futures) | usdm-futures | 97.6% | 0.096 | MAGMAUSDT | 100.0% | 1 | 1.6M | 139.3% |
| 36 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 97.4% | 0.096 | STABLEUSDT | 100.0% | 2 | 18.7M | 42.5% |
| 37 | GPS | GPSUSDT (spot) | spot, usdm-futures | 97.4% | 0.126 | EPICUSDT | 100.0% | 2 | 431.5K | 104.3% |
| 38 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 97.2% | 0.113 | BTCUSDT | 100.0% | 2 | 2.9M | 148.8% |
| 39 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 97.1% | 0.117 | HDUSDT | 100.0% | 1 | 506.9K | 24.9% |
| 40 | UB | UBUSDT (usdm-futures) | usdm-futures | 97.0% | 0.113 | USUSDT | 100.0% | 1 | 19.9M | 261.3% |
| 41 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 96.9% | 0.099 | EDGEUSDT | 100.0% | 1 | 1.9M | 116.6% |
| 42 | G | GUSDT (spot) | spot, usdm-futures | 96.8% | 0.153 | BTCUSDT | 100.0% | 3 | 559.4K | 178.8% |
| 43 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 96.8% | 0.102 | PIVXUSDT | 100.0% | 1 | 40.4M | 273.4% |
| 44 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 96.7% | 0.153 | BTCUSDT | 100.0% | 2 | 2.2M | 166.1% |
| 45 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.6% | 0.125 | BTCUSDT | 100.0% | 2 | 1.4M | 171.0% |
| 46 | GLMR | GLMRUSDT (spot) | spot | 96.5% | 0.183 | BTCUSDT | 100.0% | 12 | 318.6K | 136.8% |
| 47 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 96.3% | 0.102 | MANTAUSDT | 100.0% | 1 | 22.7M | 316.2% |
| 48 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 96.2% | 0.138 | BTCUSDT | 100.0% | 1 | 2.9M | 173.4% |
| 49 | DKNG | DKNGUSDT (usdm-futures) | usdm-futures | 96.0% | 0.126 | ADBEUSDT | 100.0% | 3 | 271.2K | 56.9% |
| 50 | SYN | SYNUSDT (spot) | spot, usdm-futures | 95.7% | 0.138 | HEIUSDT | 100.0% | 1 | 14M | 373.7% |
| 51 | APR | APRUSDT (usdm-futures) | usdm-futures | 95.7% | 0.174 | BTCUSDT | 100.0% | 2 | 3M | 121.3% |
| 52 | ON | ONUSDT (usdm-futures) | usdm-futures | 95.6% | 0.143 | BTCUSDT | 100.0% | 1 | 1.5M | 200.4% |
| 53 | PORTO | PORTOUSDT (spot) | spot | 95.5% | 0.177 | BTCUSDT | 100.0% | 3 | 215.8K | 183.4% |
| 54 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 95.4% | 0.125 | MUSDT | 100.0% | 1 | 14.2M | 264.9% |
| 55 | DGB | DGBUSDT (spot) | spot | 95.3% | 0.118 | BTCUSDT | 100.0% | 5 | 80.3K | 169.9% |
| 56 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 95.3% | 0.122 | XPINUSDT | 100.0% | 1 | 9.4M | 283.3% |
| 57 | AIN | AINUSDT (usdm-futures) | usdm-futures | 95.3% | 0.157 | BTCUSDT | 100.0% | 1 | 1.3M | 167.3% |
| 58 | TA | TAUSDT (usdm-futures) | usdm-futures | 95.2% | 0.117 | AGTUSDT | 100.0% | 1 | 1.6M | 111.8% |
| 59 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 94.9% | 0.146 | BTCUSDT | 100.0% | 3 | 1.5M | 166.3% |
| 60 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 94.8% | 0.153 | BTCUSDT | 100.0% | 1 | 725K | 188.0% |
| 61 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 94.7% | 0.141 | SKYAIUSDT | 100.0% | 1 | 2M | 203.4% |
| 62 | PROM | PROMUSDT (spot) | spot, usdm-futures | 94.6% | 0.156 | DEXEUSDT | 100.0% | 2 | 229.4K | 157.3% |
| 63 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 94.5% | 0.122 | BTCUSDT | 100.0% | 1 | 20.1M | 428.3% |
| 64 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 94.4% | 0.157 | BTCUSDT | 100.0% | 1 | 51.8M | 268.4% |
| 65 | RPL | RPLUSDT (spot) | spot, usdm-futures | 94.2% | 0.173 | TLMUSDT | 100.0% | 9 | 272K | 174.0% |
| 66 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 94.2% | 0.201 | BTCUSDT | 100.0% | 3 | 1.1M | 142.4% |
| 67 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 94.2% | 0.195 | BTCUSDT | 100.0% | 2 | 197.4K | 33.0% |
| 68 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 93.9% | 0.150 | TAIKOUSDT | 100.0% | 1 | 3.3M | 186.1% |
| 69 | XLE | XLEUSDT (usdm-futures) | usdm-futures | 93.9% | 0.136 | EDGEUSDT | 100.0% | 2 | 164.1K | 27.2% |
| 70 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 93.6% | 0.121 | BTCUSDT | 100.0% | 1 | 4.5M | 159.2% |
| 71 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 93.5% | 0.203 | BTCUSDT | 100.0% | 1 | 1.7M | 56.1% |
| 72 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 93.4% | 0.153 | USUSDT | 100.0% | 1 | 3.3M | 190.1% |
| 73 | PUNDIX | PUNDIXUSDT (spot) | spot, usdm-futures | 93.3% | 0.177 | BTCUSDT | 100.0% | 2 | 228.6K | 149.0% |
| 74 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 93.2% | 0.244 | BTCUSDT | 100.0% | 2 | 234.2K | 157.7% |
| 75 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 93.0% | 0.185 | BTCUSDT | 100.0% | 1 | 1.9M | 140.6% |
| 76 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 92.9% | 0.183 | BTCUSDT | 100.0% | 1 | 4.1M | 219.6% |
| 77 | TAC | TACUSDT (usdm-futures) | usdm-futures | 92.8% | 0.184 | ADBEUSDT | 100.0% | 1 | 12.8M | 854.7% |
| 78 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 92.7% | 0.171 | TAIKOUSDT | 100.0% | 1 | 45.5M | 396.0% |
| 79 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 92.6% | 0.253 | BTCUSDT | 100.0% | 1 | 2M | 125.0% |
| 80 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 92.5% | 0.183 | BTCUSDT | 100.0% | 2 | 1.6M | 154.2% |
| 81 | STAR | STARUSDT (usdm-futures) | usdm-futures | 92.5% | 0.188 | BTCUSDT | 100.0% | 0 | 1.8M | 186.5% |
| 82 | HOT | HOTUSDT (spot) | spot, usdm-futures | 92.3% | 0.226 | BTCUSDT | 100.0% | 6 | 323.5K | 154.1% |
| 83 | 4 | 4USDT (usdm-futures) | usdm-futures | 92.0% | 0.227 | BTCUSDT | 100.0% | 1 | 2.2M | 146.2% |
| 84 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 91.8% | 0.169 | ESPORTSUSDT | 100.0% | 2 | 2M | 118.6% |
| 85 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 91.7% | 0.202 | BTCUSDT | 100.0% | 1 | 8.1M | 229.2% |
| 86 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 91.7% | 0.132 | ALCHUSDT | 100.0% | 2 | 14M | 248.4% |
| 87 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 91.6% | 0.228 | BTCUSDT | 100.0% | 1 | 1.6M | 199.7% |
| 88 | BEL | BELUSDT (spot) | spot, usdm-futures | 91.6% | 0.205 | BTCUSDT | 100.0% | 1 | 1.3M | 184.9% |
| 89 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 91.5% | 0.192 | ALLOUSDT | 100.0% | 2 | 941.3K | 478.7% |
| 90 | GNO | GNOUSDT (spot) | spot | 91.4% | 0.255 | BTCUSDT | 100.0% | 2 | 108.5K | 65.7% |
| 91 | ID | IDUSDT (spot) | spot, usdm-futures | 91.3% | 0.192 | BTCUSDT | 100.0% | 2 | 1.1M | 137.1% |
| 92 | OGN | OGNUSDT (spot) | spot, usdm-futures | 91.3% | 0.266 | BTCUSDT | 100.0% | 2 | 927.9K | 160.0% |
| 93 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 91.2% | 0.148 | AINUSDT | 100.0% | 2 | 1M | 195.7% |
| 94 | BAR | BARUSDT (spot) | spot | 91.1% | 0.192 | ATMUSDT | 100.0% | 3 | 459.4K | 106.9% |
| 95 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 90.9% | 0.174 | BTCUSDT | 100.0% | 2 | 1.2M | 99.9% |
| 96 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 90.8% | 0.265 | ADBEUSDT | 100.0% | 2 | 1.1M | 47.9% |
| 97 | BAN | BANUSDT (usdm-futures) | usdm-futures | 90.8% | 0.205 | BTCUSDT | 100.0% | 2 | 2.2M | 99.2% |
| 98 | AT | ATUSDT (spot) | spot, usdm-futures | 90.6% | 0.202 | BTCUSDT | 100.0% | 2 | 317.8K | 88.6% |
| 99 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 90.5% | 0.158 | EPICUSDT | 100.0% | 1 | 2.7M | 289.2% |
| 100 | ONG | ONGUSDT (spot) | spot, usdm-futures | 90.3% | 0.236 | BTCUSDT | 100.0% | 1 | 226.6K | 121.7% |
| 101 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 90.2% | 0.167 | BTCUSDT | 100.0% | 1 | 2.3M | 148.1% |
| 102 | AI | AIUSDT (spot) | spot | 90.2% | 0.204 | BTCUSDT | 100.0% | 5 | 314.8K | 147.9% |
| 103 | ACE | ACEUSDT (spot) | spot, usdm-futures | 89.9% | 0.229 | BTCUSDT | 100.0% | 2 | 318.9K | 189.4% |
| 104 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 89.9% | 0.130 | XLEUSDT | 100.0% | 1 | 3.2M | 182.9% |
| 105 | CC | CCUSDT (usdm-futures) | usdm-futures | 89.8% | 0.222 | BTCUSDT | 100.0% | 1 | 4M | 61.2% |
| 106 | PYR | PYRUSDT (spot) | spot | 89.8% | 0.186 | XPINUSDT | 100.0% | 6 | 893.9K | 208.8% |
| 107 | Q | QUSDT (usdm-futures) | usdm-futures | 89.6% | 0.154 | BTCUSDT | 100.0% | 1 | 1.1M | 102.5% |
| 108 | JST | JSTUSDT (spot) | spot, usdm-futures | 89.6% | 0.165 | BTCUSDT | 100.0% | 1 | 2.7M | 43.6% |
| 109 | ERA | ERAUSDT (spot) | spot, usdm-futures | 89.4% | 0.292 | BTCUSDT | 100.0% | 3 | 229.7K | 174.8% |
| 110 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 89.4% | 0.196 | BTCUSDT | 100.0% | 0 | 13.9M | 191.7% |
| 111 | ACT | ACTUSDT (spot) | spot, usdm-futures | 89.3% | 0.246 | BTCUSDT | 100.0% | 3 | 672.3K | 201.9% |
| 112 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 89.1% | 0.158 | DKNGUSDT | 100.0% | 1 | 400.4K | 35.5% |
| 113 | THE | THEUSDT (spot) | spot, usdm-futures | 89.0% | 0.218 | BTCUSDT | 100.0% | 2 | 681.7K | 207.4% |
| 114 | MMT | MMTUSDT (spot) | spot, usdm-futures | 89.0% | 0.167 | LUMIAUSDT | 100.0% | 1 | 1.1M | 141.5% |
| 115 | XEC | XECUSDT (spot) | spot, usdm-futures | 88.8% | 0.206 | BTCUSDT | 100.0% | 2 | 317.1K | 164.7% |
| 116 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 88.7% | 0.246 | BTCUSDT | 100.0% | 2 | 1.9M | 121.4% |
| 117 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 88.4% | 0.240 | BTCUSDT | 100.0% | 1 | 4.4M | 162.3% |
| 118 | SKL | SKLUSDT (spot) | spot, usdm-futures | 88.4% | 0.285 | BTCUSDT | 100.0% | 4 | 545.3K | 153.1% |
| 119 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 88.3% | 0.190 | BTCUSDT | 100.0% | 2 | 1.4M | 112.6% |
| 120 | OPG | OPGUSDT (spot) | spot, usdm-futures | 88.1% | 0.227 | BTCUSDT | 100.0% | 2 | 1.7M | 175.2% |
| 121 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 88.0% | 0.154 | BTCUSDT | 100.0% | 1 | 1.9M | 199.3% |
| 122 | FF | FFUSDT (spot) | spot, usdm-futures | 87.8% | 0.169 | BTCUSDT | 100.0% | 1 | 1.2M | 73.1% |
| 123 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 87.7% | 0.249 | BTCUSDT | 100.0% | 1 | 2.5M | 155.8% |
| 124 | XAN | XANUSDT (usdm-futures) | usdm-futures | 87.6% | 0.242 | BTCUSDT | 100.0% | 0 | 2.9M | 164.4% |
| 125 | QUICK | QUICKUSDT (spot) | spot | 87.4% | 0.194 | BRUSDT | 100.0% | 2 | 103.3K | 163.4% |
| 126 | AWE | AWEUSDT (spot) | spot, usdm-futures | 87.4% | 0.174 | BTCUSDT | 100.0% | 1 | 390.4K | 108.3% |
| 127 | SXT | SXTUSDT (spot) | spot, usdm-futures | 87.3% | 0.237 | BTCUSDT | 100.0% | 2 | 718.2K | 156.2% |
| 128 | BANK | BANKUSDT (spot) | spot, usdm-futures | 87.1% | 0.257 | DEXEUSDT | 100.0% | 3 | 477.4K | 432.1% |
| 129 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 86.9% | 0.235 | BTCUSDT | 100.0% | 1 | 499.3K | 87.7% |
| 130 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 86.7% | 0.205 | BTCUSDT | 100.0% | 1 | 3.3M | 187.0% |
| 131 | SUN | SUNUSDT (spot) | spot, usdm-futures | 86.6% | 0.291 | BTCUSDT | 100.0% | 3 | 666.5K | 25.8% |
| 132 | IQ | IQUSDT (spot) | spot | 86.5% | 0.253 | BTCUSDT | 100.0% | 2 | 54.4K | 74.5% |
| 133 | QKC | QKCUSDT (spot) | spot | 86.5% | 0.223 | PHAROSUSDT | 100.0% | 2 | 111.1K | 144.8% |
| 134 | IN | INUSDT (usdm-futures) | usdm-futures | 86.4% | 0.219 | CCUSDT | 100.0% | 1 | 6.5M | 300.4% |
| 135 | SENT | SENTUSDT (spot) | spot, usdm-futures | 86.3% | 0.251 | BTCUSDT | 100.0% | 2 | 1.4M | 116.3% |
| 136 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 86.1% | 0.174 | BTCUSDT | 100.0% | 4 | 900.6K | 126.6% |
| 137 | RIF | RIFUSDT (spot) | spot, usdm-futures | 86.0% | 0.183 | DEXEUSDT | 100.0% | 2 | 2M | 252.9% |
| 138 | LAB | LABUSDT (usdm-futures) | usdm-futures | 85.9% | 0.193 | TACUSDT | 100.0% | 0 | 389.8M | 521.9% |
| 139 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 85.8% | 0.206 | BTCUSDT | 100.0% | 1 | 16.1M | 205.5% |
| 140 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 85.7% | 0.218 | BTCUSDT | 100.0% | 2 | 938.2K | 122.5% |
| 141 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 85.4% | 0.178 | BTCUSDT | 100.0% | 1 | 1.4M | 164.5% |
| 142 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 85.3% | 0.278 | BTCUSDT | 100.0% | 1 | 21.5M | 71.8% |
| 143 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 85.1% | 0.293 | BTCUSDT | 100.0% | 2 | 338.9K | 134.2% |
| 144 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 84.9% | 0.229 | BTCUSDT | 100.0% | 1 | 9.5M | 222.3% |
| 145 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 84.8% | 0.151 | BTCUSDT | 100.0% | 1 | 1.1M | 142.6% |
| 146 | GME | GMEUSDT (usdm-futures) | usdm-futures | 84.6% | 0.210 | BTCUSDT | 100.0% | 2 | 175.7K | 31.5% |
| 147 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 84.2% | 0.276 | BTCUSDT | 100.0% | 2 | 475.6K | 154.2% |
| 148 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 84.2% | 0.191 | UBUSDT | 100.0% | 1 | 1.5M | 214.2% |
| 149 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 84.2% | 0.194 | GPSUSDT | 100.0% | 1 | 3.5M | 217.1% |
| 150 | NOM | NOMUSDT (spot) | spot, usdm-futures | 84.0% | 0.282 | BTCUSDT | 100.0% | 5 | 600.4K | 170.0% |
| 151 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 83.9% | 0.214 | BTCUSDT | 100.0% | 2 | 3.4M | 154.3% |
| 152 | KITE | KITEUSDT (spot) | spot, usdm-futures | 83.8% | 0.286 | BTCUSDT | 100.0% | 1 | 4M | 121.3% |
| 153 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 83.5% | 0.274 | BTCUSDT | 100.0% | 1 | 745.8K | 97.3% |
| 154 | POWR | POWRUSDT (spot) | spot, usdm-futures | 83.3% | 0.326 | BTCUSDT | 100.0% | 3 | 157.5K | 130.4% |
| 155 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 83.1% | 0.327 | BTCUSDT | 100.0% | 2 | 1.2M | 128.7% |
| 156 | TUT | TUTUSDT (spot) | spot, usdm-futures | 83.0% | 0.285 | BTCUSDT | 100.0% | 2 | 457.8K | 103.3% |
| 157 | FTT | FTTUSDT (spot) | spot | 82.9% | 0.320 | BTCUSDT | 100.0% | 2 | 200.6K | 122.2% |
| 158 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 82.9% | 0.281 | BTCUSDT | 100.0% | 2 | 1.3M | 117.3% |
| 159 | ZKP | ZKPUSDT (spot) | spot, usdm-futures | 82.8% | 0.271 | BTCUSDT | 100.0% | 3 | 335.2K | 131.2% |
| 160 | BX | BXUSDT (usdm-futures) | usdm-futures | 82.7% | 0.178 | BTCUSDT | 100.0% | 1 | 573.5K | 47.6% |
| 161 | DCR | DCRUSDT (spot) | spot | 82.4% | 0.274 | BTCUSDT | 100.0% | 2 | 198.7K | 128.0% |
| 162 | JTO | JTOUSDT (spot) | spot, usdm-futures | 82.4% | 0.285 | BTCUSDT | 100.0% | 1 | 5.4M | 129.7% |
| 163 | STRAX | STRAXUSDT (spot) | spot | 82.1% | 0.256 | BTCUSDT | 100.0% | 2 | 494.5K | 126.7% |
| 164 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 82.1% | 0.246 | BROCCOLIF3BUSDT | 100.0% | 1 | 4.1M | 243.5% |
| 165 | RE | REUSDT (spot) | spot, usdm-futures | 82.1% | 0.257 | BTCUSDT | 100.0% | 1 | 20.8M | 180.3% |
| 166 | T | TUSDT (spot) | spot, usdm-futures | 81.9% | 0.289 | BTCUSDT | 100.0% | 4 | 468.8K | 153.7% |
| 167 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 81.8% | 0.245 | EDGEUSDT | 100.0% | 1 | 2.8M | 327.3% |
| 168 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 81.5% | 0.301 | BTCUSDT | 100.0% | 1 | 604.9K | 68.5% |
| 169 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 81.5% | 0.327 | BTCUSDT | 100.0% | 1 | 1.2M | 120.1% |
| 170 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 81.4% | 0.238 | CRWDUSDT | 100.0% | 3 | 379.2K | 26.6% |
| 171 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 81.1% | 0.354 | BTCUSDT | 100.0% | 1 | 776.7K | 83.4% |
| 172 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 81.0% | 0.201 | GMEUSDT | 100.0% | 4 | 222.8K | 33.6% |
| 173 | ONE | ONEUSDT (spot) | spot, usdm-futures | 80.9% | 0.352 | BTCUSDT | 100.0% | 6 | 194.4K | 153.2% |
| 174 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 80.9% | 0.217 | BTCUSDT | 100.0% | 1 | 2M | 96.6% |
| 175 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 80.6% | 0.271 | BTCUSDT | 100.0% | 2 | 2.4M | 85.8% |
| 176 | AMP | AMPUSDT (spot) | spot | 80.6% | 0.256 | BTCUSDT | 100.0% | 5 | 382.5K | 95.4% |
| 177 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 80.5% | 0.291 | BTCUSDT | 100.0% | 2 | 1.3M | 136.6% |
| 178 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 80.2% | 0.290 | BTCUSDT | 100.0% | 5 | 722.4K | 153.3% |
| 179 | REQ | REQUSDT (spot) | spot | 80.1% | 0.401 | BTCUSDT | 100.0% | 5 | 87.8K | 69.5% |
| 180 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 79.9% | 0.332 | BTCUSDT | 100.0% | 1 | 1.7M | 137.8% |
| 181 | MBL | MBLUSDT (spot) | spot | 79.6% | 0.255 | QKCUSDT | 100.0% | 3 | 660.2K | 97.9% |
| 182 | STG | STGUSDT (spot) | spot, usdm-futures | 79.6% | 0.278 | BTCUSDT | 100.0% | 2 | 733.6K | 120.7% |
| 183 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 79.4% | 0.371 | BTCUSDT | 100.0% | 4 | 618.1K | 106.2% |
| 184 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 79.2% | 0.273 | BTCUSDT | 100.0% | 1 | 3M | 187.2% |
| 185 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 79.1% | 0.304 | BTCUSDT | 100.0% | 0 | 1.7M | 157.7% |
| 186 | V | VUSDT (usdm-futures) | usdm-futures | 79.0% | 0.210 | HDUSDT | 100.0% | 3 | 292K | 30.9% |
| 187 | QI | QIUSDT (spot) | spot | 79.0% | 0.386 | PIVXUSDT | 100.0% | 10 | 140.9K | 136.0% |
| 188 | AUDIO | AUDIOUSDT (spot) | spot | 78.8% | 0.283 | BTCUSDT | 100.0% | 2 | 295.9K | 105.0% |
| 189 | GUN | GUNUSDT (spot) | spot, usdm-futures | 78.8% | 0.380 | BTCUSDT | 100.0% | 3 | 678.4K | 125.7% |
| 190 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 78.7% | 0.375 | BTCUSDT | 100.0% | 2 | 3M | 127.1% |
| 191 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 78.4% | 0.440 | BTCUSDT | 100.0% | 2 | 948.8K | 123.2% |
| 192 | LIT | LITUSDT (usdm-futures) | usdm-futures | 78.3% | 0.378 | BTCUSDT | 100.0% | 1 | 49.7M | 138.7% |
| 193 | COPPER | COPPERUSDT (usdm-futures) | usdm-futures | 78.0% | 0.382 | BTCUSDT | 100.0% | 2 | 4.9M | 23.8% |
| 194 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 77.9% | 0.365 | BTCUSDT | 100.0% | 3 | 2M | 97.1% |
| 195 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 77.7% | 0.380 | BTCUSDT | 100.0% | 1 | 1.2M | 128.8% |
| 196 | NMR | NMRUSDT (spot) | spot, usdm-futures | 77.5% | 0.372 | BTCUSDT | 100.0% | 3 | 303.5K | 74.7% |
| 197 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 77.4% | 0.323 | BTCUSDT | 100.0% | 2 | 664.1K | 104.3% |
| 198 | SKY | SKYUSDT (spot) | spot, usdm-futures | 77.2% | 0.459 | BTCUSDT | 100.0% | 1 | 1.2M | 74.8% |
| 199 | HYPER | HYPERUSDT (spot) | spot, usdm-futures | 77.1% | 0.390 | BTCUSDT | 100.0% | 3 | 1.1M | 88.4% |
| 200 | CITY | CITYUSDT (spot) | spot | 76.5% | 0.295 | BARUSDT | 100.0% | 3 | 755.5K | 119.8% |
| 201 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 76.4% | 0.229 | IDOLUSDT | 100.0% | 1 | 721.8K | 79.1% |
| 202 | TREE | TREEUSDT (spot) | spot, usdm-futures | 76.3% | 0.366 | BTCUSDT | 100.0% | 3 | 604.3K | 129.1% |
| 203 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 76.2% | 0.248 | CRWDUSDT | 100.0% | 3 | 8.3M | 315.8% |
| 204 | XPL | XPLUSDT (spot) | spot, usdm-futures | 75.8% | 0.424 | BTCUSDT | 100.0% | 1 | 9.8M | 125.7% |
| 205 | MITO | MITOUSDT (spot) | spot, usdm-futures | 75.5% | 0.290 | BTCUSDT | 100.0% | 2 | 784.5K | 121.6% |
| 206 | HEMI | HEMIUSDT (spot) | spot, usdm-futures | 75.4% | 0.282 | TOWNSUSDT | 100.0% | 2 | 789.1K | 159.9% |
| 207 | H | HUSDT (usdm-futures) | usdm-futures | 75.2% | 0.323 | MUSDT | 100.0% | 1 | 12.2M | 281.4% |
| 208 | CELO | CELOUSDT (spot) | spot, usdm-futures | 75.0% | 0.350 | BTCUSDT | 100.0% | 2 | 951.7K | 115.4% |
| 209 | TST | TSTUSDT (spot) | spot, usdm-futures | 74.8% | 0.307 | BTCUSDT | 100.0% | 2 | 595.4K | 111.4% |
| 210 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 74.6% | 0.369 | BTCUSDT | 100.0% | 2 | 674.3K | 101.2% |
| 211 | GTC | GTCUSDT (spot) | spot, usdm-futures | 74.3% | 0.367 | BTCUSDT | 100.0% | 9 | 126.2K | 122.8% |
| 212 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 74.3% | 0.445 | BTCUSDT | 100.0% | 2 | 522.8K | 99.3% |
| 213 | POPCAT | POPCATUSDT (usdm-futures) | usdm-futures | 74.2% | 0.450 | BTCUSDT | 100.0% | 1 | 2.1M | 121.1% |
| 214 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 74.1% | 0.293 | BXUSDT | 100.0% | 2 | 383.7K | 28.8% |
| 215 | WIN | WINUSDT (spot) | spot | 74.0% | 0.468 | BTCUSDT | 100.0% | 3 | 98.2K | 54.4% |
| 216 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 73.8% | 0.357 | BTCUSDT | 100.0% | 2 | 705K | 220.6% |
| 217 | ARK | ARKUSDT (spot) | spot, usdm-futures | 73.8% | 0.401 | BTCUSDT | 100.0% | 2 | 56.9K | 116.9% |
| 218 | GNS | GNSUSDT (spot) | spot | 73.6% | 0.410 | BTCUSDT | 100.0% | 5 | 64.4K | 49.3% |
| 219 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 73.5% | 0.375 | BTCUSDT | 100.0% | 1 | 1.2M | 110.8% |
| 220 | MET | METUSDT (spot) | spot, usdm-futures | 73.5% | 0.382 | BTCUSDT | 100.0% | 1 | 1.1M | 134.2% |
| 221 | OPN | OPNUSDT (spot) | spot, usdm-futures | 73.2% | 0.314 | BTCUSDT | 100.0% | 3 | 4.8M | 128.8% |
| 222 | BOB | 1000000BOBUSDT (usdm-futures) | usdm-futures | 73.1% | 0.380 | BTCUSDT | 100.0% | 2 | 632.1K | 88.1% |
| 223 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 73.0% | 0.349 | BTCUSDT | 100.0% | 1 | 1.7M | 106.6% |
| 224 | ELSA | ELSAUSDT (usdm-futures) | usdm-futures | 72.9% | 0.324 | BTCUSDT | 100.0% | 1 | 2.7M | 111.6% |
| 225 | KAT | KATUSDT (spot) | spot, usdm-futures | 72.7% | 0.383 | BTCUSDT | 100.0% | 2 | 900.3K | 102.7% |
| 226 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 72.5% | 0.341 | BTCUSDT | 100.0% | 2 | 14.3M | 218.2% |
| 227 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 72.5% | 0.399 | BTCUSDT | 100.0% | 1 | 3.7M | 109.6% |
| 228 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 72.4% | 0.272 | BTCUSDT | 100.0% | 6 | 468.4K | 75.1% |
| 229 | C | CUSDT (spot) | spot, usdm-futures | 72.3% | 0.384 | BTCUSDT | 100.0% | 3 | 332K | 86.6% |
| 230 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 72.2% | 0.202 | HDUSDT | 100.0% | 2 | 336.3K | 41.4% |
| 231 | BZ | BZUSDT (usdm-futures) | usdm-futures | 72.1% | 0.406 | XLEUSDT | 100.0% | 2 | 119.2M | 44.6% |
| 232 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 72.1% | 0.384 | BTCUSDT | 100.0% | 1 | 682K | 91.8% |
| 233 | GRASS | GRASSUSDT (usdm-futures) | usdm-futures | 72.1% | 0.384 | BTCUSDT | 100.0% | 2 | 10.3M | 155.2% |
| 234 | RED | REDUSDT (spot) | spot, usdm-futures | 71.9% | 0.448 | BTCUSDT | 100.0% | 3 | 495.2K | 99.8% |
| 235 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 71.7% | 0.395 | BTCUSDT | 100.0% | 1 | 1.1M | 98.8% |
| 236 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 71.2% | 0.274 | BTCUSDT | 100.0% | 1 | 12.2M | 160.2% |
| 237 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 71.2% | 0.469 | BTCUSDT | 100.0% | 5 | 622K | 70.1% |
| 238 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 71.1% | 0.314 | BTCUSDT | 100.0% | 2 | 7.4M | 195.1% |
| 239 | HFT | HFTUSDT (spot) | spot, usdm-futures | 70.8% | 0.317 | BTCUSDT | 100.0% | 5 | 180.5K | 138.2% |
| 240 | APE | APEUSDT (spot) | spot, usdm-futures | 70.7% | 0.401 | BTCUSDT | 100.0% | 2 | 1.1M | 94.2% |
| 241 | RARE | RAREUSDT (spot) | spot, usdm-futures | 70.5% | 0.390 | BTCUSDT | 100.0% | 12 | 201.4K | 95.6% |
| 242 | BBX | BBXUSDT (usdm-futures) | usdm-futures | 70.5% | 0.248 | BXUSDT | 100.0% | 2 | 2.7M | 116.9% |
| 243 | WET | WETUSDT (usdm-futures) | usdm-futures | 70.3% | 0.361 | BTCUSDT | 100.0% | 1 | 1.8M | 107.9% |
| 244 | VANA | VANAUSDT (spot) | spot, usdm-futures | 70.1% | 0.434 | BTCUSDT | 100.0% | 3 | 638.4K | 80.3% |
| 245 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 69.7% | 0.447 | BTCUSDT | 100.0% | 3 | 3.6M | 54.1% |
| 246 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 69.6% | 0.344 | BTCUSDT | 100.0% | 1 | 4.5M | 147.1% |
| 247 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 69.5% | 0.347 | HFTUSDT | 100.0% | 9 | 283.1K | 138.0% |
| 248 | LA | LAUSDT (spot) | spot, usdm-futures | 69.3% | 0.403 | ERAUSDT | 100.0% | 2 | 639.6K | 123.0% |
| 249 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 69.2% | 0.359 | DKNGUSDT | 100.0% | 2 | 330.4K | 39.5% |
| 250 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 69.1% | 0.315 | BTCUSDT | 100.0% | 2 | 876.5K | 115.0% |
| 251 | ADX | ADXUSDT (spot) | spot | 69.1% | 0.425 | BTCUSDT | 100.0% | 3 | 233.2K | 63.1% |
| 252 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 68.9% | 0.402 | BTCUSDT | 100.0% | 2 | 265.2K | 80.2% |
| 253 | CHILLGUY | CHILLGUYUSDT (usdm-futures) | usdm-futures | 68.7% | 0.454 | BTCUSDT | 100.0% | 1 | 903.6K | 115.6% |
| 254 | ESP | ESPUSDT (spot) | spot, usdm-futures | 68.5% | 0.400 | INUSDT | 100.0% | 1 | 523K | 93.0% |
| 255 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 68.4% | 0.445 | BTCUSDT | 100.0% | 2 | 919.2K | 98.0% |
| 256 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 68.4% | 0.234 | BTCUSDT | 100.0% | 3 | 435.2K | 207.3% |
| 257 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 68.4% | 0.402 | BTCUSDT | 100.0% | 2 | 1.8M | 75.2% |
| 258 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 68.3% | 0.305 | COPPERUSDT | 100.0% | 1 | 1.5M | 80.0% |
| 259 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 67.9% | 0.461 | BTCUSDT | 100.0% | 2 | 330.2K | 86.3% |
| 260 | SC | SCUSDT (spot) | spot | 67.7% | 0.456 | BTCUSDT | 100.0% | 5 | 169.8K | 63.7% |
| 261 | HOLO | HOLOUSDT (spot) | spot, usdm-futures | 67.5% | 0.374 | BTCUSDT | 100.0% | 2 | 804.8K | 93.9% |
| 262 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 67.4% | 0.321 | UBERUSDT | 100.0% | 1 | 9.2M | 33.1% |
| 263 | BREV | BREVUSDT (spot) | spot, usdm-futures | 67.3% | 0.330 | WETUSDT | 100.0% | 3 | 311.7K | 139.5% |
| 264 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 67.2% | 0.347 | BTCUSDT | 100.0% | 1 | 2.5M | 132.5% |
| 265 | MINA | MINAUSDT (spot) | spot, usdm-futures | 67.1% | 0.478 | BTCUSDT | 100.0% | 2 | 392.1K | 92.5% |
| 266 | BICO | BICOUSDT (spot) | spot, usdm-futures | 67.1% | 0.328 | BTCUSDT | 100.0% | 1 | 915.9K | 151.3% |
| 267 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 66.9% | 0.483 | BTCUSDT | 100.0% | 4 | 396.5K | 89.6% |
| 268 | YB | YBUSDT (spot) | spot, usdm-futures | 66.7% | 0.395 | BTCUSDT | 100.0% | 2 | 295.3K | 109.9% |
| 269 | DIS | DISUSDT (usdm-futures) | usdm-futures | 66.7% | 0.376 | HDUSDT | 100.0% | 2 | 224.5K | 27.5% |
| 270 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 66.6% | 0.450 | BTCUSDT | 100.0% | 1 | 1.6M | 114.1% |
| 271 | HPE | HPEUSDT (usdm-futures) | usdm-futures | 66.3% | 0.326 | SPCXBUSDT | 100.0% | 1 | 735.8K | 70.4% |
| 272 | MEME | MEMEUSDT (spot) | spot, usdm-futures | 65.9% | 0.419 | BTCUSDT | 100.0% | 3 | 567.4K | 84.8% |
| 273 | XTZ | XTZUSDT (spot) | spot, usdm-futures | 65.9% | 0.539 | BTCUSDT | 100.0% | 1 | 405.7K | 78.7% |
| 274 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 65.8% | 0.391 | BTCUSDT | 100.0% | 1 | 2.9M | 117.0% |
| 275 | RAD | RADUSDT (spot) | spot | 65.6% | 0.391 | BTCUSDT | 100.0% | 7 | 216.5K | 55.1% |
| 276 | FLOCK | FLOCKUSDT (usdm-futures) | usdm-futures | 65.5% | 0.459 | BTCUSDT | 100.0% | 1 | 1.3M | 117.2% |
| 277 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 65.4% | 0.493 | BTCUSDT | 100.0% | 5 | 189.5K | 82.2% |
| 278 | ALICE | ALICEUSDT (spot) | spot, usdm-futures | 65.2% | 0.437 | ACEUSDT | 100.0% | 2 | 1.1M | 129.5% |
| 279 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 65.0% | 0.450 | BTCUSDT | 100.0% | 1 | 119K | 67.8% |
| 280 | A | AUSDT (spot) | spot, usdm-futures | 64.7% | 0.472 | BTCUSDT | 100.0% | 2 | 440.6K | 82.9% |
| 281 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 64.6% | 0.343 | AMZNUSDT | 100.0% | 1 | 17M | 33.9% |
| 282 | KAS | KASUSDT (usdm-futures) | usdm-futures | 64.5% | 0.512 | BTCUSDT | 100.0% | 1 | 2.9M | 63.8% |

## Diagnostics

- Basis size selected: 282
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.081
- Maximum pairwise absolute correlation: 0.539
- Mean whole-market projection R²: 83.6%
- Median whole-market projection R²: 80.2%
- 10th-percentile whole-market projection R²: 62.6%
- Minimum whole-market projection R²: 58.5%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.6% | 18.1% | 1.4% | 0.0% |
| 5 | 24.0% | 19.6% | 2.4% | 0.1% |
| 10 | 25.6% | 21.5% | 3.6% | 0.8% |
| 15 | 27.0% | 22.5% | 4.5% | 1.3% |
| 20 | 28.5% | 23.8% | 5.5% | 2.2% |
| 25 | 30.0% | 24.9% | 6.4% | 2.9% |
| 30 | 31.5% | 26.6% | 7.4% | 3.5% |
| 35 | 33.0% | 28.0% | 8.6% | 5.0% |
| 40 | 34.3% | 29.3% | 9.8% | 6.0% |
| 45 | 35.8% | 30.9% | 10.8% | 6.9% |
| 50 | 37.2% | 32.1% | 11.9% | 8.5% |
| 55 | 38.6% | 33.2% | 13.1% | 9.2% |
| 60 | 39.9% | 34.8% | 14.3% | 10.3% |
| 65 | 41.2% | 35.8% | 15.1% | 11.3% |
| 70 | 42.7% | 37.0% | 16.5% | 12.5% |
| 75 | 44.3% | 38.5% | 17.6% | 13.7% |
| 80 | 45.7% | 40.1% | 18.9% | 14.4% |
| 85 | 46.9% | 41.2% | 19.7% | 16.0% |
| 90 | 48.1% | 42.1% | 21.1% | 16.6% |
| 95 | 49.4% | 43.0% | 22.3% | 17.6% |
| 100 | 50.8% | 44.7% | 23.6% | 18.6% |
| 105 | 52.1% | 46.1% | 25.0% | 19.4% |
| 110 | 53.4% | 47.6% | 26.0% | 20.2% |
| 115 | 54.6% | 48.5% | 27.2% | 21.4% |
| 120 | 55.8% | 49.7% | 28.0% | 22.5% |
| 125 | 56.8% | 50.6% | 29.6% | 23.7% |
| 130 | 57.8% | 52.0% | 30.6% | 25.0% |
| 135 | 58.9% | 52.8% | 31.5% | 25.9% |
| 140 | 60.0% | 53.9% | 32.4% | 27.0% |
| 145 | 61.1% | 54.7% | 34.0% | 28.5% |
| 150 | 62.2% | 55.9% | 35.0% | 29.7% |
| 155 | 63.2% | 56.7% | 36.1% | 31.1% |
| 160 | 64.2% | 58.0% | 38.0% | 32.0% |
| 165 | 65.2% | 59.1% | 39.2% | 33.0% |
| 170 | 66.3% | 60.0% | 40.5% | 34.2% |
| 175 | 67.2% | 60.7% | 41.5% | 35.1% |
| 180 | 68.1% | 61.6% | 42.5% | 36.6% |
| 185 | 69.1% | 63.0% | 43.3% | 37.5% |
| 190 | 70.0% | 63.9% | 44.4% | 38.6% |
| 195 | 71.1% | 64.8% | 45.6% | 39.9% |
| 200 | 72.0% | 66.0% | 46.7% | 41.7% |
| 205 | 72.9% | 67.4% | 47.7% | 43.2% |
| 210 | 73.6% | 68.1% | 48.7% | 44.8% |
| 215 | 74.4% | 69.0% | 49.9% | 45.5% |
| 220 | 75.2% | 70.0% | 50.8% | 46.5% |
| 225 | 75.9% | 70.7% | 51.8% | 47.4% |
| 230 | 76.7% | 71.5% | 53.6% | 48.0% |
| 235 | 77.4% | 72.2% | 54.6% | 49.2% |
| 240 | 78.1% | 73.2% | 55.4% | 50.2% |
| 245 | 78.9% | 74.0% | 56.5% | 51.5% |
| 250 | 79.6% | 74.9% | 57.3% | 52.3% |
| 255 | 80.2% | 75.7% | 58.0% | 53.2% |
| 260 | 80.9% | 76.6% | 59.0% | 54.4% |
| 265 | 81.5% | 77.6% | 59.7% | 54.9% |
| 270 | 82.2% | 78.4% | 60.7% | 56.1% |
| 275 | 82.8% | 79.4% | 61.6% | 57.1% |
| 280 | 83.4% | 79.8% | 62.3% | 58.2% |
| 282 | 83.6% | 80.2% | 62.6% | 58.5% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | ATM | TAIKO | HD | DEXE | US | MAGMA | HEI | XNO | ADBE | EVAA | U | AKE | ZEREBRO | KGST | BTTC | PIVX | DODO | BAS | B | HANA | M | MANTA | ALCH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.005 | 0.015 | -0.015 | -0.016 | 0.020 | 0.025 | 0.006 | 0.040 | 0.035 | 0.046 | 0.022 | 0.064 | 0.052 | -0.035 | 0.088 | 0.056 | 0.106 | 0.058 | 0.022 | 0.080 | 0.074 | 0.116 | 0.090 |
| ATM | 0.005 | 1.000 | -0.005 | 0.006 | -0.010 | -0.012 | -0.021 | -0.036 | -0.004 | -0.030 | 0.030 | -0.032 | 0.011 | 0.023 | -0.015 | 0.028 | 0.030 | -0.026 | 0.066 | -0.027 | 0.001 | 0.021 | -0.045 | 0.008 |
| TAIKO | 0.015 | -0.005 | 1.000 | -0.006 | 0.000 | -0.003 | 0.022 | -0.025 | -0.018 | -0.042 | 0.022 | -0.055 | 0.019 | -0.004 | 0.007 | 0.012 | -0.015 | 0.014 | 0.026 | -0.011 | -0.011 | 0.042 | -0.001 | -0.005 |
| HD | -0.015 | 0.006 | -0.006 | 1.000 | -0.003 | -0.010 | -0.016 | -0.027 | -0.030 | 0.023 | 0.008 | -0.037 | 0.007 | 0.021 | 0.001 | 0.008 | 0.011 | -0.022 | 0.040 | 0.047 | 0.021 | 0.047 | -0.016 | 0.007 |
| DEXE | -0.016 | -0.010 | 0.000 | -0.003 | 1.000 | 0.025 | 0.010 | 0.023 | 0.045 | 0.003 | 0.017 | -0.009 | -0.020 | -0.002 | 0.065 | -0.040 | 0.021 | -0.011 | 0.005 | 0.048 | -0.006 | -0.013 | -0.009 | -0.010 |
| US | 0.020 | -0.012 | -0.003 | -0.010 | 0.025 | 1.000 | 0.028 | -0.014 | 0.022 | -0.008 | 0.029 | -0.007 | 0.060 | -0.044 | 0.027 | 0.011 | 0.000 | 0.003 | 0.013 | 0.031 | 0.054 | 0.041 | 0.002 | 0.037 |
| MAGMA | 0.025 | -0.021 | 0.022 | -0.016 | 0.010 | 0.028 | 1.000 | 0.018 | -0.023 | 0.041 | -0.032 | -0.031 | -0.011 | -0.011 | -0.067 | -0.000 | -0.053 | 0.019 | 0.001 | 0.001 | 0.020 | -0.024 | 0.000 | -0.000 |
| HEI | 0.006 | -0.036 | -0.025 | -0.027 | 0.023 | -0.014 | 0.018 | 1.000 | 0.002 | -0.019 | 0.028 | 0.035 | -0.017 | -0.042 | 0.024 | 0.021 | 0.043 | 0.008 | 0.017 | -0.019 | -0.022 | 0.003 | 0.012 | 0.009 |
| XNO | 0.040 | -0.004 | -0.018 | -0.030 | 0.045 | 0.022 | -0.023 | 0.002 | 1.000 | 0.024 | 0.030 | 0.024 | -0.012 | 0.036 | -0.002 | -0.006 | 0.005 | 0.020 | 0.028 | 0.028 | 0.048 | 0.014 | 0.009 | 0.004 |
| ADBE | 0.035 | -0.030 | -0.042 | 0.023 | 0.003 | -0.008 | 0.041 | -0.019 | 0.024 | 1.000 | 0.014 | 0.033 | 0.030 | -0.008 | 0.008 | -0.027 | 0.005 | -0.003 | 0.015 | 0.002 | 0.041 | -0.007 | -0.013 | -0.009 |
| EVAA | 0.046 | 0.030 | 0.022 | 0.008 | 0.017 | 0.029 | -0.032 | 0.028 | 0.030 | 0.014 | 1.000 | 0.006 | 0.012 | 0.051 | 0.019 | -0.013 | 0.007 | 0.002 | 0.009 | 0.095 | 0.019 | -0.005 | 0.033 | 0.012 |
| U | 0.022 | -0.032 | -0.055 | -0.037 | -0.009 | -0.007 | -0.031 | 0.035 | 0.024 | 0.033 | 0.006 | 1.000 | -0.009 | 0.007 | 0.011 | -0.025 | 0.043 | 0.048 | -0.040 | 0.006 | 0.071 | 0.073 | -0.008 | 0.074 |
| AKE | 0.064 | 0.011 | 0.019 | 0.007 | -0.020 | 0.060 | -0.011 | -0.017 | -0.012 | 0.030 | 0.012 | -0.009 | 1.000 | 0.002 | -0.010 | 0.004 | 0.012 | -0.009 | 0.001 | 0.027 | 0.057 | 0.010 | 0.036 | -0.054 |
| ZEREBRO | 0.052 | 0.023 | -0.004 | 0.021 | -0.002 | -0.044 | -0.011 | -0.042 | 0.036 | -0.008 | 0.051 | 0.007 | 0.002 | 1.000 | -0.022 | -0.010 | -0.046 | 0.009 | -0.011 | 0.036 | 0.025 | 0.004 | 0.015 | 0.036 |
| KGST | -0.035 | -0.015 | 0.007 | 0.001 | 0.065 | 0.027 | -0.067 | 0.024 | -0.002 | 0.008 | 0.019 | 0.011 | -0.010 | -0.022 | 1.000 | 0.012 | -0.007 | -0.008 | -0.021 | 0.013 | 0.009 | 0.008 | -0.028 | -0.003 |
| BTTC | 0.088 | 0.028 | 0.012 | 0.008 | -0.040 | 0.011 | -0.000 | 0.021 | -0.006 | -0.027 | -0.013 | -0.025 | 0.004 | -0.010 | 0.012 | 1.000 | -0.013 | 0.008 | -0.045 | 0.020 | 0.003 | -0.005 | 0.046 | 0.064 |
| PIVX | 0.056 | 0.030 | -0.015 | 0.011 | 0.021 | 0.000 | -0.053 | 0.043 | 0.005 | 0.005 | 0.007 | 0.043 | 0.012 | -0.046 | -0.007 | -0.013 | 1.000 | 0.018 | 0.006 | 0.005 | -0.004 | 0.060 | 0.028 | 0.006 |
| DODO | 0.106 | -0.026 | 0.014 | -0.022 | -0.011 | 0.003 | 0.019 | 0.008 | 0.020 | -0.003 | 0.002 | 0.048 | -0.009 | 0.009 | -0.008 | 0.008 | 0.018 | 1.000 | -0.017 | 0.001 | 0.003 | 0.053 | 0.036 | 0.045 |
| BAS | 0.058 | 0.066 | 0.026 | 0.040 | 0.005 | 0.013 | 0.001 | 0.017 | 0.028 | 0.015 | 0.009 | -0.040 | 0.001 | -0.011 | -0.021 | -0.045 | 0.006 | -0.017 | 1.000 | -0.037 | 0.020 | -0.020 | -0.031 | -0.000 |
| B | 0.022 | -0.027 | -0.011 | 0.047 | 0.048 | 0.031 | 0.001 | -0.019 | 0.028 | 0.002 | 0.095 | 0.006 | 0.027 | 0.036 | 0.013 | 0.020 | 0.005 | 0.001 | -0.037 | 1.000 | 0.011 | 0.013 | 0.033 | 0.003 |
| HANA | 0.080 | 0.001 | -0.011 | 0.021 | -0.006 | 0.054 | 0.020 | -0.022 | 0.048 | 0.041 | 0.019 | 0.071 | 0.057 | 0.025 | 0.009 | 0.003 | -0.004 | 0.003 | 0.020 | 0.011 | 1.000 | 0.002 | 0.041 | 0.043 |
| M | 0.074 | 0.021 | 0.042 | 0.047 | -0.013 | 0.041 | -0.024 | 0.003 | 0.014 | -0.007 | -0.005 | 0.073 | 0.010 | 0.004 | 0.008 | -0.005 | 0.060 | 0.053 | -0.020 | 0.013 | 0.002 | 1.000 | -0.006 | 0.029 |
| MANTA | 0.116 | -0.045 | -0.001 | -0.016 | -0.009 | 0.002 | 0.000 | 0.012 | 0.009 | -0.013 | 0.033 | -0.008 | 0.036 | 0.015 | -0.028 | 0.046 | 0.028 | 0.036 | -0.031 | 0.033 | 0.041 | -0.006 | 1.000 | 0.023 |
| ALCH | 0.090 | 0.008 | -0.005 | 0.007 | -0.010 | 0.037 | -0.000 | 0.009 | 0.004 | -0.009 | 0.012 | 0.074 | -0.054 | 0.036 | -0.003 | 0.064 | 0.006 | 0.045 | -0.000 | 0.003 | 0.043 | 0.029 | 0.023 | 1.000 |

The complete 282 × 282 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| SNX | SNXUSDT | 58.5% | 64.4% | BTCUSDT | 0.518 |
| 2Z | 2ZUSDT | 58.6% | 64.4% | BTCUSDT | 0.483 |
| FLUID | FLUIDUSDT | 58.6% | 64.3% | BTCUSDT | 0.493 |
| SPORTFUN | SPORTFUNUSDT | 58.9% | 64.1% | BTCUSDT | 0.354 |
| PYTH | PYTHUSDT | 58.9% | 64.1% | BTCUSDT | 0.455 |
| SPACE | SPACEUSDT | 59.0% | 64.1% | BTCUSDT | 0.413 |
| XMR | XMRUSDT | 59.0% | 64.0% | BTCUSDT | 0.436 |
| INX | INXUSDT | 59.1% | 64.0% | BTCUSDT | 0.419 |
| PUMP | PUMPUSDT | 59.2% | 63.9% | BTCUSDT | 0.525 |
| USTC | USTCUSDT | 59.3% | 63.8% | BTCUSDT | 0.462 |
| FLNC | FLNCUSDT | 59.3% | 63.8% | BTCUSDT | 0.356 |
| VELODROME | VELODROMEUSDT | 59.3% | 63.8% | BTCUSDT | 0.507 |
| BABY | BABYUSDT | 59.4% | 63.7% | BTCUSDT | 0.477 |
| AIA | AIAUSDT | 59.4% | 63.7% | BTCUSDT | 0.496 |
| LIGHT | LIGHTUSDT | 59.5% | 63.6% | BTCUSDT | 0.424 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

