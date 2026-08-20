# Binance portfolio basis

Generated 2026-07-23T18:33:57.973Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-06-23T00:00Z through 2026-07-22T20:00Z
- Sampling: 4h log returns (180 samples over 30 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 638 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.2B | 39.2% |
| 2 | HEI | HEIUSDT (spot) | spot, usdm-futures | 100.0% | 0.001 | BTCUSDT | 100.0% | 4 | 4.1M | 245.3% |
| 3 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 100.0% | 0.019 | BTCUSDT | 100.0% | 0 | 9.4M | 271.0% |
| 4 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 100.0% | 0.024 | HEIUSDT | 100.0% | 4 | 1.7M | 264.5% |
| 5 | V | VUSDT (usdm-futures) | usdm-futures | 99.9% | 0.026 | CLOUSDT | 100.0% | 4 | 292K | 28.7% |
| 6 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 99.8% | 0.043 | VUSDT | 100.0% | 0 | 22.7M | 284.1% |
| 7 | DODOX | DODOXUSDT (usdm-futures) | usdm-futures | 99.6% | 0.054 | HEIUSDT | 100.0% | 0 | 3.9M | 221.0% |
| 8 | M | MUSDT (usdm-futures) | usdm-futures | 99.5% | 0.072 | HEIUSDT | 100.0% | 0 | 12.4M | 559.7% |
| 9 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 99.4% | 0.066 | BTCUSDT | 100.0% | 0 | 1.9M | 110.0% |
| 10 | XNO | XNOUSDT (spot) | spot | 99.3% | 0.094 | BTCUSDT | 100.0% | 8 | 42.3K | 172.1% |
| 11 | PYR | PYRUSDT (spot) | spot | 99.0% | 0.090 | BTCUSDT | 100.0% | 16 | 893.9K | 182.2% |
| 12 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 98.8% | 0.083 | VUSDT | 100.0% | 4 | 1.4M | 172.4% |
| 13 | BR | BRUSDT (usdm-futures) | usdm-futures | 98.7% | 0.081 | DODOXUSDT | 100.0% | 0 | 1.6M | 139.7% |
| 14 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 98.5% | 0.100 | BTCUSDT | 100.0% | 0 | 4.5M | 154.9% |
| 15 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 98.2% | 0.095 | ZESTUSDT | 100.0% | 4 | 3.5M | 561.1% |
| 16 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 97.8% | 0.114 | BTCUSDT | 100.0% | 0 | 1.2M | 188.2% |
| 17 | IN | INUSDT (usdm-futures) | usdm-futures | 97.6% | 0.097 | BRUSDT | 100.0% | 0 | 6.5M | 479.5% |
| 18 | XEC | XECUSDT (spot) | spot, usdm-futures | 97.6% | 0.122 | BTCUSDT | 100.0% | 8 | 317.1K | 165.2% |
| 19 | BEL | BELUSDT (spot) | spot, usdm-futures | 96.9% | 0.145 | HEIUSDT | 100.0% | 4 | 1.3M | 208.0% |
| 20 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 96.9% | 0.151 | BTCUSDT | 100.0% | 4 | 1.7M | 49.4% |
| 21 | KGST | KGSTUSDT (spot) | spot | 96.6% | 0.132 | PYRUSDT | 100.0% | 20 | 87.2K | 2.9% |
| 22 | BAR | BARUSDT (spot) | spot | 96.6% | 0.125 | DODOXUSDT | 100.0% | 8 | 459.4K | 93.0% |
| 23 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 96.0% | 0.193 | CLOUSDT | 100.0% | 0 | 8.3M | 442.6% |
| 24 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 96.0% | 0.192 | ZESTUSDT | 100.0% | 0 | 9.5M | 248.9% |
| 25 | TAC | TACUSDT (usdm-futures) | usdm-futures | 95.7% | 0.154 | CLOUSDT | 100.0% | 4 | 12.8M | 864.0% |
| 26 | Q | QUSDT (usdm-futures) | usdm-futures | 95.4% | 0.134 | EPICUSDT | 100.0% | 4 | 1.1M | 97.9% |
| 27 | BREV | BREVUSDT (spot) | spot, usdm-futures | 95.0% | 0.147 | TAIKOUSDT | 100.0% | 4 | 311.7K | 150.3% |
| 28 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 94.8% | 0.213 | BTCUSDT | 100.0% | 0 | 1.7M | 143.3% |
| 29 | B | BUSDT (usdm-futures) | usdm-futures | 94.2% | 0.149 | PYRUSDT | 100.0% | 0 | 5.1M | 360.7% |
| 30 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 93.8% | 0.197 | BELUSDT | 100.0% | 4 | 234.2K | 183.9% |
| 31 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 93.7% | 0.196 | VUSDT | 100.0% | 0 | 941.3K | 483.5% |
| 32 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 93.2% | 0.124 | HOMEUSDT | 100.0% | 0 | 3.3M | 165.0% |
| 33 | SKL | SKLUSDT (spot) | spot, usdm-futures | 93.1% | 0.213 | BTCUSDT | 100.0% | 4 | 545.3K | 163.3% |
| 34 | DGB | DGBUSDT (spot) | spot | 92.8% | 0.222 | XNOUSDT | 100.0% | 8 | 80.3K | 147.2% |
| 35 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 92.5% | 0.200 | BTCUSDT | 100.0% | 0 | 1.6M | 206.1% |
| 36 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 92.4% | 0.206 | XNOUSDT | 100.0% | 4 | 435.2K | 197.3% |
| 37 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 92.3% | 0.165 | HOMEUSDT | 100.0% | 4 | 3.6M | 472.8% |
| 38 | TLM | TLMUSDT (spot) | spot, usdm-futures | 92.1% | 0.174 | BTCUSDT | 100.0% | 4 | 3.8M | 406.9% |
| 39 | ACT | ACTUSDT (spot) | spot, usdm-futures | 92.0% | 0.215 | 币安人生USDT | 100.0% | 4 | 672.3K | 239.0% |
| 40 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 91.4% | 0.159 | TACUSDT | 100.0% | 4 | 18.7M | 37.6% |
| 41 | AWE | AWEUSDT (spot) | spot, usdm-futures | 91.3% | 0.183 | HEIUSDT | 100.0% | 4 | 390.4K | 91.6% |
| 42 | RPL | RPLUSDT (spot) | spot, usdm-futures | 90.8% | 0.223 | BTCUSDT | 100.0% | 16 | 272K | 179.8% |
| 43 | H | HUSDT (usdm-futures) | usdm-futures | 90.0% | 0.160 | BRUSDT | 100.0% | 0 | 12.2M | 224.5% |
| 44 | QUICK | QUICKUSDT (spot) | spot | 89.9% | 0.288 | MUSDT | 100.0% | 4 | 103.3K | 129.0% |
| 45 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 89.6% | 0.249 | CLOUSDT | 100.0% | 4 | 1.9M | 112.8% |
| 46 | BAS | BASUSDT (usdm-futures) | usdm-futures | 89.4% | 0.216 | ARPAUSDT | 100.0% | 0 | 6.6M | 349.6% |
| 47 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 88.6% | 0.269 | 币安人生USDT | 100.0% | 0 | 812K | 242.1% |
| 48 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 88.6% | 0.181 | ZESTUSDT | 100.0% | 0 | 14M | 339.7% |
| 49 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 88.1% | 0.203 | PYRUSDT | 100.0% | 0 | 2.3M | 151.7% |
| 50 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 88.1% | 0.232 | ZESTUSDT | 100.0% | 4 | 2.8M | 433.6% |
| 51 | PORTO | PORTOUSDT (spot) | spot | 87.5% | 0.233 | AKEUSDT | 100.0% | 4 | 215.8K | 162.9% |
| 52 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 87.4% | 0.293 | BTCUSDT | 100.0% | 4 | 1.2M | 123.9% |
| 53 | LLY | LLYUSDT (usdm-futures) | usdm-futures | 86.8% | 0.187 | VUSDT | 100.0% | 0 | 1.1M | 39.1% |
| 54 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 86.6% | 0.230 | QUSDT | 100.0% | 0 | 24M | 439.4% |
| 55 | ONG | ONGUSDT (spot) | spot, usdm-futures | 85.6% | 0.201 | MUSDT | 100.0% | 4 | 226.6K | 102.7% |
| 56 | QKC | QKCUSDT (spot) | spot | 85.3% | 0.240 | CLOUSDT | 100.0% | 4 | 111.1K | 121.0% |
| 57 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 85.2% | 0.176 | INUSDT | 100.0% | 0 | 1.6M | 173.9% |
| 58 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 85.0% | 0.298 | AVAAIUSDT | 100.0% | 4 | 1.1M | 141.2% |
| 59 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 84.7% | 0.170 | DEXEUSDT | 100.0% | 4 | 20.1M | 442.8% |
| 60 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 84.3% | 0.253 | CRWDUSDT | 100.0% | 0 | 7.1M | 210.8% |
| 61 | 4 | 4USDT (usdm-futures) | usdm-futures | 84.2% | 0.214 | ZESTUSDT | 100.0% | 4 | 2.2M | 150.1% |
| 62 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 84.1% | 0.167 | PRLUSDT | 100.0% | 0 | 1.1M | 125.6% |
| 63 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 83.8% | 0.213 | BTCUSDT | 100.0% | 4 | 338.9K | 104.1% |
| 64 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 83.1% | 0.204 | MAGMAUSDT | 100.0% | 0 | 2.7M | 283.2% |
| 65 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 82.7% | 0.216 | XECUSDT | 100.0% | 4 | 3.4M | 92.8% |
| 66 | SXT | SXTUSDT (spot) | spot, usdm-futures | 82.7% | 0.215 | DODOXUSDT | 100.0% | 4 | 718.2K | 149.1% |
| 67 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 82.5% | 0.201 | BREVUSDT | 100.0% | 4 | 14M | 263.2% |
| 68 | ONE | ONEUSDT (spot) | spot, usdm-futures | 82.3% | 0.313 | BTCUSDT | 100.0% | 12 | 194.4K | 145.9% |
| 69 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 81.9% | 0.310 | BTCUSDT | 100.0% | 0 | 2.5M | 64.3% |
| 70 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 81.1% | 0.380 | BTCUSDT | 100.0% | 4 | 948.8K | 112.2% |
| 71 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 80.4% | 0.277 | BTCUSDT | 100.0% | 4 | 265.2K | 75.3% |
| 72 | CATI | CATIUSDT (spot) | spot, usdm-futures | 79.9% | 0.192 | TACUSDT | 100.0% | 4 | 385.8K | 100.8% |
| 73 | ID | IDUSDT (spot) | spot, usdm-futures | 78.4% | 0.204 | AIOUSDT | 100.0% | 12 | 1.1M | 132.0% |
| 74 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 78.1% | 0.216 | HUSDT | 100.0% | 0 | 40.4M | 273.1% |
| 75 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 77.9% | 0.245 | INUSDT | 100.0% | 4 | 876.5K | 94.4% |
| 76 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 77.8% | 0.268 | ESPORTSUSDT | 100.0% | 8 | 1.2M | 125.5% |
| 77 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 77.4% | 0.249 | ARIAUSDT | 100.0% | 4 | 1.4M | 175.4% |
| 78 | JST | JSTUSDT (spot) | spot, usdm-futures | 77.0% | 0.309 | BILLUSDT | 100.0% | 4 | 2.7M | 46.4% |
| 79 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 76.7% | 0.243 | TAGUSDT | 100.0% | 4 | 499.3K | 84.6% |
| 80 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 76.0% | 0.204 | TACUSDT | 100.0% | 0 | 2M | 233.5% |
| 81 | PROM | PROMUSDT (spot) | spot, usdm-futures | 76.0% | 0.264 | DEXEUSDT | 100.0% | 4 | 229.4K | 189.5% |
| 82 | BZ | BZUSDT (usdm-futures) | usdm-futures | 75.2% | 0.233 | HANAUSDT | 100.0% | 0 | 119.2M | 43.4% |
| 83 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 75.0% | 0.452 | BTCUSDT | 100.0% | 12 | 396.5K | 83.2% |
| 84 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 74.6% | 0.277 | 4USDT | 100.0% | 4 | 2M | 111.6% |
| 85 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 74.3% | 0.405 | PYRUSDT | 100.0% | 4 | 2.9M | 124.5% |
| 86 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 73.1% | 0.276 | FOGOUSDT | 100.0% | 4 | 197.4K | 29.2% |
| 87 | TA | TAUSDT (usdm-futures) | usdm-futures | 72.0% | 0.227 | BARUSDT | 100.0% | 4 | 1.6M | 115.8% |
| 88 | SMCI | SMCIUSDT (usdm-futures) | usdm-futures | 71.9% | 0.293 | MIRAUSDT | 100.0% | 4 | 876K | 99.4% |
| 89 | ATM | ATMUSDT (spot) | spot | 71.2% | 0.365 | BARUSDT | 100.0% | 4 | 1.7M | 216.7% |
| 90 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 70.6% | 0.178 | TACUSDT | 100.0% | 0 | 3.3M | 157.8% |
| 91 | HOT | HOTUSDT (spot) | spot, usdm-futures | 69.8% | 0.263 | BTCUSDT | 100.0% | 8 | 323.5K | 99.5% |
| 92 | ERA | ERAUSDT (spot) | spot, usdm-futures | 69.5% | 0.333 | XNOUSDT | 100.0% | 4 | 229.7K | 197.0% |
| 93 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 69.5% | 0.292 | ARIAUSDT | 100.0% | 4 | 1.2M | 105.0% |
| 94 | RE | REUSDT (spot) | spot, usdm-futures | 69.2% | 0.253 | BTCUSDT | 100.0% | 4 | 20.8M | 179.8% |
| 95 | U | UUSDT (spot) | spot | 69.1% | 0.196 | FOGOUSDT | 100.0% | 24 | 15.6M | 0.5% |
| 96 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 68.6% | 0.375 | LLYUSDT | 100.0% | 4 | 1.1M | 52.7% |
| 97 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 68.4% | 0.247 | HMSTRUSDT | 100.0% | 4 | 8.1M | 236.0% |
| 98 | KITE | KITEUSDT (spot) | spot, usdm-futures | 67.6% | 0.331 | BTCUSDT | 100.0% | 4 | 4M | 116.1% |
| 99 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 67.2% | 0.400 | BTCUSDT | 100.0% | 4 | 919.2K | 83.4% |
| 100 | AT | ATUSDT (spot) | spot, usdm-futures | 66.9% | 0.239 | ARPAUSDT | 100.0% | 4 | 317.8K | 80.0% |
| 101 | MINA | MINAUSDT (spot) | spot, usdm-futures | 66.8% | 0.428 | BTCUSDT | 100.0% | 4 | 392.1K | 88.4% |
| 102 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 66.3% | 0.348 | ACTUSDT | 100.0% | 4 | 14.3M | 228.2% |
| 103 | POPCAT | POPCATUSDT (usdm-futures) | usdm-futures | 65.7% | 0.291 | BTCUSDT | 100.0% | 4 | 2.1M | 110.6% |
| 104 | RIF | RIFUSDT (spot) | spot, usdm-futures | 65.6% | 0.237 | RPLUSDT | 100.0% | 4 | 2M | 287.7% |
| 105 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 65.0% | 0.267 | MAVIAUSDT | 100.0% | 0 | 14.2M | 243.5% |
| 106 | GPS | GPSUSDT (spot) | spot, usdm-futures | 64.6% | 0.212 | TLMUSDT | 100.0% | 4 | 431.5K | 102.4% |
| 107 | QNT | QNTUSDT (spot) | spot, usdm-futures | 64.5% | 0.492 | BTCUSDT | 100.0% | 4 | 597.8K | 49.7% |
| 108 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 64.0% | 0.227 | POWERUSDT | 100.0% | 0 | 4.4M | 157.9% |
| 109 | ARK | ARKUSDT (spot) | spot, usdm-futures | 63.4% | 0.318 | BELUSDT | 100.0% | 4 | 56.9K | 127.5% |
| 110 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 62.9% | 0.353 | QKCUSDT | 100.0% | 4 | 1.7M | 96.1% |
| 111 | SKY | SKYUSDT (spot) | spot, usdm-futures | 61.4% | 0.363 | BTCUSDT | 100.0% | 4 | 1.2M | 77.5% |
| 112 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 61.0% | 0.275 | SAFEUSDT | 100.0% | 4 | 1M | 200.9% |
| 113 | G | GUSDT (spot) | spot, usdm-futures | 60.6% | 0.265 | HUMAUSDT | 100.0% | 8 | 559.4K | 148.5% |
| 114 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 60.6% | 0.189 | PHAROSUSDT | 100.0% | 0 | 3.5M | 139.0% |
| 115 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 60.3% | 0.313 | SAHARAUSDT | 100.0% | 8 | 8.6M | 127.7% |

## Diagnostics

- Basis size selected: 115
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.063
- Maximum pairwise absolute correlation: 0.492
- Mean whole-market projection R²: 82.0%
- Median whole-market projection R²: 80.3%
- 10th-percentile whole-market projection R²: 70.6%
- Minimum whole-market projection R²: 64.7%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.8% | 14.1% | 0.7% | 0.0% |
| 5 | 22.9% | 17.2% | 3.6% | 0.4% |
| 10 | 26.0% | 20.6% | 6.3% | 1.9% |
| 15 | 29.5% | 24.0% | 9.3% | 4.4% |
| 20 | 32.9% | 27.2% | 12.3% | 6.6% |
| 25 | 36.2% | 31.6% | 15.2% | 9.0% |
| 30 | 40.3% | 35.3% | 19.1% | 12.2% |
| 35 | 43.1% | 38.7% | 22.2% | 14.6% |
| 40 | 46.2% | 42.2% | 25.2% | 16.7% |
| 45 | 48.9% | 45.2% | 28.7% | 20.0% |
| 50 | 51.7% | 47.4% | 31.5% | 23.4% |
| 55 | 54.8% | 50.8% | 34.7% | 27.3% |
| 60 | 57.3% | 53.8% | 38.0% | 29.2% |
| 65 | 60.1% | 56.4% | 41.1% | 31.7% |
| 70 | 62.7% | 58.8% | 44.4% | 35.3% |
| 75 | 65.1% | 61.5% | 47.0% | 39.4% |
| 80 | 67.3% | 63.9% | 50.1% | 42.3% |
| 85 | 70.0% | 66.9% | 53.5% | 46.5% |
| 90 | 72.2% | 69.4% | 56.6% | 51.2% |
| 95 | 74.2% | 71.7% | 59.5% | 52.9% |
| 100 | 76.1% | 73.4% | 62.1% | 55.4% |
| 105 | 78.2% | 75.7% | 64.8% | 58.3% |
| 110 | 80.2% | 78.1% | 67.2% | 62.3% |
| 115 | 82.0% | 80.3% | 70.6% | 64.7% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | HEI | CLO | EPIC | V | SKYAI | DODOX | M | ZEST | XNO | PYR | HOME | BR | 币安人生 | TAIKO | ALCH | IN | XEC | BEL | ANTHROPIC | KGST | BAR | DEXE | BILL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.001 | 0.019 | -0.007 | 0.016 | 0.008 | -0.019 | 0.014 | 0.066 | 0.094 | 0.090 | 0.039 | 0.073 | 0.100 | -0.031 | 0.114 | -0.064 | 0.122 | 0.028 | 0.151 | 0.066 | 0.115 | 0.004 | 0.103 |
| HEI | -0.001 | 1.000 | -0.010 | 0.024 | -0.018 | -0.034 | -0.054 | 0.072 | 0.005 | 0.018 | 0.016 | -0.056 | 0.028 | -0.031 | 0.022 | -0.065 | 0.013 | -0.014 | -0.145 | -0.011 | 0.074 | -0.058 | -0.024 | 0.061 |
| CLO | 0.019 | -0.010 | 1.000 | -0.003 | -0.026 | -0.026 | -0.039 | 0.034 | 0.026 | 0.014 | 0.060 | 0.057 | 0.001 | -0.046 | 0.029 | -0.065 | 0.014 | -0.006 | 0.064 | -0.028 | 0.131 | 0.022 | 0.193 | -0.059 |
| EPIC | -0.007 | 0.024 | -0.003 | 1.000 | -0.002 | -0.020 | 0.041 | 0.021 | -0.049 | -0.032 | -0.019 | -0.019 | -0.015 | 0.059 | 0.015 | -0.014 | 0.001 | -0.053 | -0.054 | 0.036 | 0.090 | -0.015 | -0.024 | -0.023 |
| V | 0.016 | -0.018 | -0.026 | -0.002 | 1.000 | -0.043 | -0.029 | 0.038 | 0.043 | -0.039 | 0.020 | -0.083 | -0.065 | 0.014 | -0.076 | 0.037 | 0.056 | -0.059 | -0.034 | 0.045 | 0.015 | 0.018 | 0.033 | 0.004 |
| SKYAI | 0.008 | -0.034 | -0.026 | -0.020 | -0.043 | 1.000 | 0.010 | -0.038 | -0.053 | 0.023 | 0.024 | 0.054 | -0.041 | 0.061 | 0.058 | 0.014 | 0.083 | -0.001 | 0.077 | 0.037 | -0.035 | 0.077 | -0.020 | -0.005 |
| DODOX | -0.019 | -0.054 | -0.039 | 0.041 | -0.029 | 0.010 | 1.000 | 0.008 | -0.014 | -0.011 | 0.062 | -0.008 | -0.081 | -0.044 | -0.012 | 0.026 | 0.081 | -0.056 | 0.059 | -0.057 | 0.034 | -0.125 | 0.011 | -0.038 |
| M | 0.014 | 0.072 | 0.034 | 0.021 | 0.038 | -0.038 | 0.008 | 1.000 | 0.033 | 0.015 | 0.008 | -0.015 | -0.052 | 0.020 | 0.050 | 0.028 | -0.018 | -0.000 | -0.014 | -0.010 | 0.009 | -0.041 | -0.018 | 0.060 |
| ZEST | 0.066 | 0.005 | 0.026 | -0.049 | 0.043 | -0.053 | -0.014 | 0.033 | 1.000 | -0.022 | 0.027 | 0.054 | -0.009 | -0.073 | -0.095 | 0.057 | -0.035 | 0.050 | 0.050 | 0.009 | -0.002 | -0.100 | 0.010 | 0.192 |
| XNO | 0.094 | 0.018 | 0.014 | -0.032 | -0.039 | 0.023 | -0.011 | 0.015 | -0.022 | 1.000 | 0.042 | -0.023 | 0.059 | 0.011 | -0.034 | -0.013 | 0.061 | 0.011 | -0.002 | -0.108 | 0.012 | 0.055 | 0.120 | -0.067 |
| PYR | 0.090 | 0.016 | 0.060 | -0.019 | 0.020 | 0.024 | 0.062 | 0.008 | 0.027 | 0.042 | 1.000 | 0.043 | 0.002 | 0.037 | 0.065 | 0.028 | -0.009 | -0.000 | -0.003 | -0.004 | 0.132 | -0.013 | 0.062 | 0.069 |
| HOME | 0.039 | -0.056 | 0.057 | -0.019 | -0.083 | 0.054 | -0.008 | -0.015 | 0.054 | -0.023 | 0.043 | 1.000 | 0.030 | -0.017 | -0.013 | -0.055 | -0.069 | -0.097 | -0.006 | 0.015 | -0.040 | -0.055 | 0.077 | 0.033 |
| BR | 0.073 | 0.028 | 0.001 | -0.015 | -0.065 | -0.041 | -0.081 | -0.052 | -0.009 | 0.059 | 0.002 | 0.030 | 1.000 | 0.016 | -0.048 | 0.097 | -0.097 | 0.027 | -0.057 | 0.094 | -0.041 | 0.032 | -0.039 | -0.032 |
| 币安人生 | 0.100 | -0.031 | -0.046 | 0.059 | 0.014 | 0.061 | -0.044 | 0.020 | -0.073 | 0.011 | 0.037 | -0.017 | 0.016 | 1.000 | 0.070 | -0.025 | 0.045 | 0.093 | -0.088 | 0.020 | 0.069 | -0.007 | 0.082 | 0.037 |
| TAIKO | -0.031 | 0.022 | 0.029 | 0.015 | -0.076 | 0.058 | -0.012 | 0.050 | -0.095 | -0.034 | 0.065 | -0.013 | -0.048 | 0.070 | 1.000 | -0.017 | -0.065 | 0.001 | -0.039 | 0.032 | -0.016 | 0.009 | 0.011 | -0.029 |
| ALCH | 0.114 | -0.065 | -0.065 | -0.014 | 0.037 | 0.014 | 0.026 | 0.028 | 0.057 | -0.013 | 0.028 | -0.055 | 0.097 | -0.025 | -0.017 | 1.000 | -0.004 | -0.015 | 0.059 | 0.003 | -0.009 | 0.076 | -0.011 | 0.047 |
| IN | -0.064 | 0.013 | 0.014 | 0.001 | 0.056 | 0.083 | 0.081 | -0.018 | -0.035 | 0.061 | -0.009 | -0.069 | -0.097 | 0.045 | -0.065 | -0.004 | 1.000 | -0.005 | -0.030 | 0.020 | -0.008 | 0.003 | 0.010 | -0.012 |
| XEC | 0.122 | -0.014 | -0.006 | -0.053 | -0.059 | -0.001 | -0.056 | -0.000 | 0.050 | 0.011 | -0.000 | -0.097 | 0.027 | 0.093 | 0.001 | -0.015 | -0.005 | 1.000 | -0.061 | 0.056 | 0.008 | -0.043 | -0.056 | 0.083 |
| BEL | 0.028 | -0.145 | 0.064 | -0.054 | -0.034 | 0.077 | 0.059 | -0.014 | 0.050 | -0.002 | -0.003 | -0.006 | -0.057 | -0.088 | -0.039 | 0.059 | -0.030 | -0.061 | 1.000 | -0.022 | 0.056 | 0.011 | 0.015 | 0.019 |
| ANTHROPIC | 0.151 | -0.011 | -0.028 | 0.036 | 0.045 | 0.037 | -0.057 | -0.010 | 0.009 | -0.108 | -0.004 | 0.015 | 0.094 | 0.020 | 0.032 | 0.003 | 0.020 | 0.056 | -0.022 | 1.000 | 0.008 | 0.048 | -0.037 | 0.049 |
| KGST | 0.066 | 0.074 | 0.131 | 0.090 | 0.015 | -0.035 | 0.034 | 0.009 | -0.002 | 0.012 | 0.132 | -0.040 | -0.041 | 0.069 | -0.016 | -0.009 | -0.008 | 0.008 | 0.056 | 0.008 | 1.000 | -0.001 | 0.072 | 0.030 |
| BAR | 0.115 | -0.058 | 0.022 | -0.015 | 0.018 | 0.077 | -0.125 | -0.041 | -0.100 | 0.055 | -0.013 | -0.055 | 0.032 | -0.007 | 0.009 | 0.076 | 0.003 | -0.043 | 0.011 | 0.048 | -0.001 | 1.000 | 0.001 | -0.057 |
| DEXE | 0.004 | -0.024 | 0.193 | -0.024 | 0.033 | -0.020 | 0.011 | -0.018 | 0.010 | 0.120 | 0.062 | 0.077 | -0.039 | 0.082 | 0.011 | -0.011 | 0.010 | -0.056 | 0.015 | -0.037 | 0.072 | 0.001 | 1.000 | -0.010 |
| BILL | 0.103 | 0.061 | -0.059 | -0.023 | 0.004 | -0.005 | -0.038 | 0.060 | 0.192 | -0.067 | 0.069 | 0.033 | -0.032 | 0.037 | -0.029 | 0.047 | -0.012 | 0.083 | 0.019 | 0.049 | 0.030 | -0.057 | -0.010 | 1.000 |

The complete 115 × 115 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| US | USUSDT | 64.7% | 59.4% | BREVUSDT | 0.244 |
| STAR | STARUSDT | 65.0% | 59.2% | BTCUSDT | 0.265 |
| MITO | MITOUSDT | 65.2% | 59.0% | NAORISUSDT | 0.246 |
| MORPHO | MORPHOUSDT | 65.2% | 59.0% | MIRAUSDT | -0.255 |
| ETHFI | ETHFIUSDT | 65.3% | 58.9% | BTCUSDT | 0.431 |
| BULLA | BULLAUSDT | 65.8% | 58.5% | INUSDT | 0.289 |
| HD | HDUSDT | 65.8% | 58.5% | BZUSDT | -0.329 |
| VANA | VANAUSDT | 65.9% | 58.4% | EGLDUSDT | 0.363 |
| DIS | DISUSDT | 65.9% | 58.4% | VUSDT | 0.317 |
| AIOT | AIOTUSDT | 66.1% | 58.2% | ZBTUSDT | 0.238 |
| TOWNS | TOWNSUSDT | 66.4% | 58.0% | BTCUSDT | 0.315 |
| XPL | XPLUSDT | 66.4% | 57.9% | BTCUSDT | 0.381 |
| AIN | AINUSDT | 66.5% | 57.9% | HUMAUSDT | 0.272 |
| CRM | CRMUSDT | 66.7% | 57.7% | VUSDT | 0.277 |
| TRADOOR | TRADOORUSDT | 66.9% | 57.6% | RAVEUSDT | 0.267 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

