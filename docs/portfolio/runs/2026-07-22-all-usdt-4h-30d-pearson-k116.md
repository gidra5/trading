# Binance portfolio basis

Generated 2026-07-23T18:03:33.107Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-06-23T00:00Z through 2026-07-22T20:00Z
- Sampling: 4h log returns (180 samples over 30 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 128)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 624 eligible assets from 715 active continuously priced candidates
- Quality filter: at least 80.0% non-zero 4h returns and a complete window
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Median 4h quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 149.1M | 39.2% |
| 2 | HEI | HEIUSDT (spot) | spot, usdm-futures | 100.0% | 0.001 | BTCUSDT | 542.2K | 245.3% |
| 3 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 100.0% | 0.019 | BTCUSDT | 1.4M | 271.0% |
| 4 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 100.0% | 0.024 | HEIUSDT | 250.8K | 264.5% |
| 5 | V | VUSDT (usdm-futures) | usdm-futures | 99.9% | 0.026 | CLOUSDT | 27.4K | 28.7% |
| 6 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 99.8% | 0.043 | VUSDT | 3.2M | 284.1% |
| 7 | DODOX | DODOXUSDT (usdm-futures) | usdm-futures | 99.6% | 0.054 | HEIUSDT | 384.1K | 221.0% |
| 8 | M | MUSDT (usdm-futures) | usdm-futures | 99.5% | 0.072 | HEIUSDT | 1.3M | 559.7% |
| 9 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 99.4% | 0.066 | BTCUSDT | 252.7K | 110.0% |
| 10 | XNO | XNOUSDT (spot) | spot | 99.3% | 0.094 | BTCUSDT | 6.2K | 172.1% |
| 11 | PYR | PYRUSDT (spot) | spot | 99.0% | 0.090 | BTCUSDT | 99.9K | 182.2% |
| 12 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 98.8% | 0.083 | VUSDT | 247.2K | 172.4% |
| 13 | BR | BRUSDT (usdm-futures) | usdm-futures | 98.7% | 0.081 | DODOXUSDT | 212.4K | 139.7% |
| 14 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 98.5% | 0.100 | BTCUSDT | 615K | 154.9% |
| 15 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 98.2% | 0.095 | ZESTUSDT | 497.2K | 561.1% |
| 16 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 97.8% | 0.114 | BTCUSDT | 167.8K | 188.2% |
| 17 | IN | INUSDT (usdm-futures) | usdm-futures | 97.6% | 0.097 | BRUSDT | 857.3K | 479.5% |
| 18 | XEC | XECUSDT (spot) | spot, usdm-futures | 97.6% | 0.122 | BTCUSDT | 37.1K | 165.2% |
| 19 | BEL | BELUSDT (spot) | spot, usdm-futures | 96.9% | 0.145 | HEIUSDT | 197.3K | 208.0% |
| 20 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 96.9% | 0.151 | BTCUSDT | 203.7K | 49.4% |
| 21 | BAR | BARUSDT (spot) | spot | 96.6% | 0.125 | DODOXUSDT | 51.2K | 93.0% |
| 22 | Q | QUSDT (usdm-futures) | usdm-futures | 96.6% | 0.134 | EPICUSDT | 166.7K | 97.9% |
| 23 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 96.1% | 0.193 | CLOUSDT | 955.6K | 442.6% |
| 24 | B | BUSDT (usdm-futures) | usdm-futures | 95.8% | 0.149 | PYRUSDT | 689.9K | 360.7% |
| 25 | TAC | TACUSDT (usdm-futures) | usdm-futures | 95.7% | 0.154 | CLOUSDT | 1.8M | 864.0% |
| 26 | BREV | BREVUSDT (spot) | spot, usdm-futures | 95.0% | 0.147 | TAIKOUSDT | 44K | 150.3% |
| 27 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 94.7% | 0.213 | BTCUSDT | 253.8K | 143.3% |
| 28 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 94.4% | 0.192 | ZESTUSDT | 1.4M | 248.9% |
| 29 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 93.9% | 0.196 | VUSDT | 98.3K | 483.5% |
| 30 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 93.7% | 0.197 | BELUSDT | 35.1K | 183.9% |
| 31 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 93.4% | 0.124 | HOMEUSDT | 457.4K | 165.0% |
| 32 | SKL | SKLUSDT (spot) | spot, usdm-futures | 93.3% | 0.213 | BTCUSDT | 69.7K | 163.3% |
| 33 | DGB | DGBUSDT (spot) | spot | 92.9% | 0.222 | XNOUSDT | 12.6K | 147.2% |
| 34 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 92.7% | 0.147 | BTCUSDT | 415.2K | 283.2% |
| 35 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 92.6% | 0.200 | BTCUSDT | 188K | 206.1% |
| 36 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 92.3% | 0.206 | XNOUSDT | 60.9K | 197.3% |
| 37 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 92.3% | 0.165 | HOMEUSDT | 541.7K | 472.8% |
| 38 | ACT | ACTUSDT (spot) | spot, usdm-futures | 92.1% | 0.215 | 币安人生USDT | 105.4K | 239.0% |
| 39 | TLM | TLMUSDT (spot) | spot, usdm-futures | 91.4% | 0.174 | BTCUSDT | 457.4K | 406.9% |
| 40 | PIVX | PIVXUSDT (spot) | spot | 91.3% | 0.201 | SKYAIUSDT | 44.8K | 331.8% |
| 41 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 90.7% | 0.159 | TACUSDT | 1.9M | 37.6% |
| 42 | H | HUSDT (usdm-futures) | usdm-futures | 90.1% | 0.160 | BRUSDT | 1.8M | 224.5% |
| 43 | RPL | RPLUSDT (spot) | spot, usdm-futures | 90.0% | 0.223 | BTCUSDT | 41K | 179.8% |
| 44 | AI | AIUSDT (spot) | spot | 89.4% | 0.250 | BTCUSDT | 45.4K | 108.5% |
| 45 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 89.0% | 0.176 | INUSDT | 240.3K | 173.9% |
| 46 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 88.8% | 0.269 | 币安人生USDT | 123.7K | 242.1% |
| 47 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 88.4% | 0.230 | QUSDT | 3.6M | 439.4% |
| 48 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 88.2% | 0.249 | CLOUSDT | 258.9K | 112.8% |
| 49 | BAS | BASUSDT (usdm-futures) | usdm-futures | 88.1% | 0.216 | ARPAUSDT | 1.1M | 349.6% |
| 50 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 87.7% | 0.203 | PYRUSDT | 315.3K | 151.7% |
| 51 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 87.4% | 0.293 | BTCUSDT | 152.2K | 123.9% |
| 52 | PORTO | PORTOUSDT (spot) | spot | 86.9% | 0.233 | AKEUSDT | 32.8K | 162.9% |
| 53 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 86.3% | 0.277 | BTCUSDT | 41K | 75.3% |
| 54 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 86.2% | 0.213 | BTCUSDT | 48.3K | 104.1% |
| 55 | QKC | QKCUSDT (spot) | spot | 85.6% | 0.240 | CLOUSDT | 14.5K | 121.0% |
| 56 | LLY | LLYUSDT (usdm-futures) | usdm-futures | 85.2% | 0.187 | VUSDT | 103.1K | 39.1% |
| 57 | ONG | ONGUSDT (spot) | spot, usdm-futures | 85.1% | 0.201 | MUSDT | 24.6K | 102.7% |
| 58 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 84.7% | 0.201 | BREVUSDT | 1.9M | 263.2% |
| 59 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 84.6% | 0.298 | AVAAIUSDT | 141.4K | 141.2% |
| 60 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 84.4% | 0.253 | CRWDUSDT | 924.8K | 210.8% |
| 61 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 84.0% | 0.170 | DEXEUSDT | 2.9M | 442.8% |
| 62 | 4 | 4USDT (usdm-futures) | usdm-futures | 83.6% | 0.214 | ZESTUSDT | 356.1K | 150.1% |
| 63 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 83.5% | 0.232 | ZESTUSDT | 420.3K | 433.6% |
| 64 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 83.2% | 0.204 | VANRYUSDT | 1.9M | 339.7% |
| 65 | SXT | SXTUSDT (spot) | spot, usdm-futures | 82.4% | 0.215 | DODOXUSDT | 100.2K | 149.1% |
| 66 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 82.3% | 0.310 | BTCUSDT | 372.8K | 64.3% |
| 67 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 81.6% | 0.380 | BTCUSDT | 142.4K | 112.2% |
| 68 | QUICK | QUICKUSDT (spot) | spot | 81.1% | 0.357 | PIVXUSDT | 12.4K | 129.0% |
| 69 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 80.5% | 0.216 | XECUSDT | 380.5K | 92.8% |
| 70 | PROM | PROMUSDT (spot) | spot, usdm-futures | 80.0% | 0.264 | DEXEUSDT | 30.1K | 189.5% |
| 71 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 79.4% | 0.204 | TACUSDT | 342.2K | 233.5% |
| 72 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 79.0% | 0.275 | SAFEUSDT | 145.4K | 200.9% |
| 73 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 78.3% | 0.243 | TAGUSDT | 71.6K | 84.6% |
| 74 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 78.1% | 0.291 | BTCUSDT | 211.3K | 136.5% |
| 75 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 77.6% | 0.224 | BELUSDT | 514.3K | 154.0% |
| 76 | BZ | BZUSDT (usdm-futures) | usdm-futures | 77.1% | 0.233 | HANAUSDT | 18.3M | 43.4% |
| 77 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 76.8% | 0.393 | BTCUSDT | 145.5K | 92.8% |
| 78 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 76.5% | 0.277 | AIUSDT | 153.7K | 125.6% |
| 79 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 76.0% | 0.229 | BRUSDT | 231.2K | 154.5% |
| 80 | TA | TAUSDT (usdm-futures) | usdm-futures | 75.6% | 0.227 | BARUSDT | 230.9K | 115.8% |
| 81 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 75.2% | 0.452 | BTCUSDT | 59.6K | 83.2% |
| 82 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 74.3% | 0.284 | FIGHTUSDT | 177.2K | 125.5% |
| 83 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 73.8% | 0.196 | BASUSDT | 2M | 243.5% |
| 84 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 73.1% | 0.249 | ARIAUSDT | 203.9K | 175.4% |
| 85 | XAN | XANUSDT (usdm-futures) | usdm-futures | 72.7% | 0.232 | BTCUSDT | 420.4K | 142.8% |
| 86 | ID | IDUSDT (spot) | spot, usdm-futures | 72.1% | 0.204 | AIOUSDT | 189.7K | 132.0% |
| 87 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 72.1% | 0.375 | LLYUSDT | 112.8K | 52.7% |
| 88 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 71.4% | 0.255 | MIRAUSDT | 303.7K | 93.2% |
| 89 | ATM | ATMUSDT (spot) | spot | 71.3% | 0.365 | BARUSDT | 207.7K | 216.7% |
| 90 | FF | FFUSDT (spot) | spot, usdm-futures | 71.1% | 0.275 | DYDXUSDT | 174.9K | 65.5% |
| 91 | US | USUSDT (usdm-futures) | usdm-futures | 70.4% | 0.244 | BREVUSDT | 2.3M | 274.4% |
| 92 | AWE | AWEUSDT (spot) | spot, usdm-futures | 70.1% | 0.268 | PIVXUSDT | 48.3K | 91.6% |
| 93 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 69.8% | 0.178 | TACUSDT | 388.2K | 157.8% |
| 94 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 69.5% | 0.187 | BILLUSDT | 532.7K | 179.3% |
| 95 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 68.8% | 0.247 | HMSTRUSDT | 1.3M | 236.0% |
| 96 | HOT | HOTUSDT (spot) | spot, usdm-futures | 68.0% | 0.263 | BTCUSDT | 41K | 99.5% |
| 97 | HOLO | HOLOUSDT (spot) | spot, usdm-futures | 67.7% | 0.339 | BTCUSDT | 112.7K | 86.6% |
| 98 | EWJ | EWJUSDT (usdm-futures) | usdm-futures | 67.1% | 0.328 | BTCUSDT | 113.9K | 27.3% |
| 99 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 66.9% | 0.294 | TAIKOUSDT | 7.6M | 261.6% |
| 100 | BICO | BICOUSDT (spot) | spot, usdm-futures | 66.1% | 0.282 | FOLKSUSDT | 159.4K | 145.1% |
| 101 | ACE | ACEUSDT (spot) | spot, usdm-futures | 66.1% | 0.387 | ANTHROPICUSDT | 44.3K | 197.2% |
| 102 | PIPPIN | PIPPINUSDT (usdm-futures) | usdm-futures | 65.4% | 0.311 | BTCUSDT | 1.1M | 127.7% |
| 103 | ERA | ERAUSDT (spot) | spot, usdm-futures | 65.0% | 0.333 | XNOUSDT | 33.9K | 197.0% |
| 104 | T | TUSDT (spot) | spot, usdm-futures | 64.7% | 0.372 | BUSDT | 60.7K | 148.0% |
| 105 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 64.3% | 0.238 | ZBTUSDT | 249.3K | 195.1% |
| 106 | SMCI | SMCIUSDT (usdm-futures) | usdm-futures | 64.1% | 0.352 | EWJUSDT | 99.6K | 99.4% |
| 107 | BABY | BABYUSDT (spot) | spot, usdm-futures | 63.2% | 0.365 | BTCUSDT | 64.9K | 69.0% |
| 108 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 63.0% | 0.344 | BTCUSDT | 71.3K | 90.7% |
| 109 | MINA | MINAUSDT (spot) | spot, usdm-futures | 62.4% | 0.428 | BTCUSDT | 55K | 88.4% |
| 110 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 62.0% | 0.405 | PYRUSDT | 462.2K | 124.5% |
| 111 | OPN | OPNUSDT (spot) | spot, usdm-futures | 61.2% | 0.317 | PIPPINUSDT | 460.4K | 126.1% |
| 112 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 60.6% | 0.289 | INUSDT | 433.8K | 183.0% |
| 113 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 60.5% | 0.274 | BASUSDT | 667.7K | 182.5% |
| 114 | WIN | WINUSDT (spot) | spot | 60.1% | 0.340 | BTCUSDT | 14.4K | 45.7% |
| 115 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 59.8% | 0.309 | BILLUSDT | 36.7K | 30.6% |
| 116 | IQ | IQUSDT (spot) | spot | 59.7% | 0.339 | PRLUSDT | 5.9K | 64.0% |

## Diagnostics

- Basis size selected: 116
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.064
- Maximum pairwise absolute correlation: 0.452
- Mean whole-market projection R²: 82.5%
- Median whole-market projection R²: 80.5%
- 10th-percentile whole-market projection R²: 70.6%
- Minimum whole-market projection R²: 66.0%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.8% | 14.1% | 0.7% | 0.0% |
| 5 | 22.9% | 17.1% | 3.6% | 0.4% |
| 10 | 26.0% | 20.6% | 6.4% | 1.9% |
| 15 | 29.5% | 24.0% | 9.3% | 4.4% |
| 20 | 32.9% | 27.1% | 12.3% | 6.8% |
| 25 | 36.4% | 31.7% | 15.4% | 9.8% |
| 30 | 40.8% | 36.2% | 19.5% | 12.8% |
| 35 | 43.5% | 39.0% | 22.5% | 14.7% |
| 40 | 46.5% | 42.4% | 25.5% | 17.7% |
| 45 | 49.2% | 45.4% | 28.8% | 21.2% |
| 50 | 51.9% | 48.1% | 31.7% | 23.6% |
| 55 | 54.7% | 50.2% | 35.0% | 27.4% |
| 60 | 57.7% | 53.5% | 38.5% | 29.5% |
| 65 | 60.2% | 56.3% | 41.5% | 32.3% |
| 70 | 62.8% | 59.0% | 44.2% | 36.9% |
| 75 | 65.0% | 61.5% | 47.7% | 40.5% |
| 80 | 67.6% | 64.3% | 50.9% | 43.4% |
| 85 | 70.0% | 66.8% | 53.9% | 48.0% |
| 90 | 72.0% | 68.6% | 56.5% | 50.5% |
| 95 | 73.9% | 70.6% | 59.2% | 53.8% |
| 100 | 76.3% | 74.0% | 62.2% | 56.3% |
| 105 | 78.3% | 75.7% | 64.7% | 58.9% |
| 110 | 80.3% | 77.9% | 67.3% | 62.6% |
| 115 | 82.1% | 79.8% | 70.0% | 64.4% |
| 116 | 82.5% | 80.5% | 70.6% | 66.0% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | HEI | CLO | EPIC | V | SKYAI | DODOX | M | ZEST | XNO | PYR | HOME | BR | 币安人生 | TAIKO | ALCH | IN | XEC | BEL | ANTHROPIC | BAR | Q | DEXE | B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.001 | 0.019 | -0.007 | 0.016 | 0.008 | -0.019 | 0.014 | 0.066 | 0.094 | 0.090 | 0.039 | 0.073 | 0.100 | -0.031 | 0.114 | -0.064 | 0.122 | 0.028 | 0.151 | 0.115 | 0.026 | 0.004 | -0.036 |
| HEI | -0.001 | 1.000 | -0.010 | 0.024 | -0.018 | -0.034 | -0.054 | 0.072 | 0.005 | 0.018 | 0.016 | -0.056 | 0.028 | -0.031 | 0.022 | -0.065 | 0.013 | -0.014 | -0.145 | -0.011 | -0.058 | 0.015 | -0.024 | -0.061 |
| CLO | 0.019 | -0.010 | 1.000 | -0.003 | -0.026 | -0.026 | -0.039 | 0.034 | 0.026 | 0.014 | 0.060 | 0.057 | 0.001 | -0.046 | 0.029 | -0.065 | 0.014 | -0.006 | 0.064 | -0.028 | 0.022 | -0.066 | 0.193 | -0.068 |
| EPIC | -0.007 | 0.024 | -0.003 | 1.000 | -0.002 | -0.020 | 0.041 | 0.021 | -0.049 | -0.032 | -0.019 | -0.019 | -0.015 | 0.059 | 0.015 | -0.014 | 0.001 | -0.053 | -0.054 | 0.036 | -0.015 | 0.134 | -0.024 | -0.107 |
| V | 0.016 | -0.018 | -0.026 | -0.002 | 1.000 | -0.043 | -0.029 | 0.038 | 0.043 | -0.039 | 0.020 | -0.083 | -0.065 | 0.014 | -0.076 | 0.037 | 0.056 | -0.059 | -0.034 | 0.045 | 0.018 | -0.002 | 0.033 | 0.043 |
| SKYAI | 0.008 | -0.034 | -0.026 | -0.020 | -0.043 | 1.000 | 0.010 | -0.038 | -0.053 | 0.023 | 0.024 | 0.054 | -0.041 | 0.061 | 0.058 | 0.014 | 0.083 | -0.001 | 0.077 | 0.037 | 0.077 | 0.087 | -0.020 | 0.017 |
| DODOX | -0.019 | -0.054 | -0.039 | 0.041 | -0.029 | 0.010 | 1.000 | 0.008 | -0.014 | -0.011 | 0.062 | -0.008 | -0.081 | -0.044 | -0.012 | 0.026 | 0.081 | -0.056 | 0.059 | -0.057 | -0.125 | 0.094 | 0.011 | 0.016 |
| M | 0.014 | 0.072 | 0.034 | 0.021 | 0.038 | -0.038 | 0.008 | 1.000 | 0.033 | 0.015 | 0.008 | -0.015 | -0.052 | 0.020 | 0.050 | 0.028 | -0.018 | -0.000 | -0.014 | -0.010 | -0.041 | 0.008 | -0.018 | 0.040 |
| ZEST | 0.066 | 0.005 | 0.026 | -0.049 | 0.043 | -0.053 | -0.014 | 0.033 | 1.000 | -0.022 | 0.027 | 0.054 | -0.009 | -0.073 | -0.095 | 0.057 | -0.035 | 0.050 | 0.050 | 0.009 | -0.100 | 0.036 | 0.010 | -0.059 |
| XNO | 0.094 | 0.018 | 0.014 | -0.032 | -0.039 | 0.023 | -0.011 | 0.015 | -0.022 | 1.000 | 0.042 | -0.023 | 0.059 | 0.011 | -0.034 | -0.013 | 0.061 | 0.011 | -0.002 | -0.108 | 0.055 | -0.029 | 0.120 | 0.051 |
| PYR | 0.090 | 0.016 | 0.060 | -0.019 | 0.020 | 0.024 | 0.062 | 0.008 | 0.027 | 0.042 | 1.000 | 0.043 | 0.002 | 0.037 | 0.065 | 0.028 | -0.009 | -0.000 | -0.003 | -0.004 | -0.013 | -0.019 | 0.062 | -0.149 |
| HOME | 0.039 | -0.056 | 0.057 | -0.019 | -0.083 | 0.054 | -0.008 | -0.015 | 0.054 | -0.023 | 0.043 | 1.000 | 0.030 | -0.017 | -0.013 | -0.055 | -0.069 | -0.097 | -0.006 | 0.015 | -0.055 | 0.020 | 0.077 | -0.063 |
| BR | 0.073 | 0.028 | 0.001 | -0.015 | -0.065 | -0.041 | -0.081 | -0.052 | -0.009 | 0.059 | 0.002 | 0.030 | 1.000 | 0.016 | -0.048 | 0.097 | -0.097 | 0.027 | -0.057 | 0.094 | 0.032 | 0.094 | -0.039 | 0.113 |
| 币安人生 | 0.100 | -0.031 | -0.046 | 0.059 | 0.014 | 0.061 | -0.044 | 0.020 | -0.073 | 0.011 | 0.037 | -0.017 | 0.016 | 1.000 | 0.070 | -0.025 | 0.045 | 0.093 | -0.088 | 0.020 | -0.007 | 0.006 | 0.082 | 0.004 |
| TAIKO | -0.031 | 0.022 | 0.029 | 0.015 | -0.076 | 0.058 | -0.012 | 0.050 | -0.095 | -0.034 | 0.065 | -0.013 | -0.048 | 0.070 | 1.000 | -0.017 | -0.065 | 0.001 | -0.039 | 0.032 | 0.009 | -0.049 | 0.011 | -0.020 |
| ALCH | 0.114 | -0.065 | -0.065 | -0.014 | 0.037 | 0.014 | 0.026 | 0.028 | 0.057 | -0.013 | 0.028 | -0.055 | 0.097 | -0.025 | -0.017 | 1.000 | -0.004 | -0.015 | 0.059 | 0.003 | 0.076 | 0.003 | -0.011 | -0.034 |
| IN | -0.064 | 0.013 | 0.014 | 0.001 | 0.056 | 0.083 | 0.081 | -0.018 | -0.035 | 0.061 | -0.009 | -0.069 | -0.097 | 0.045 | -0.065 | -0.004 | 1.000 | -0.005 | -0.030 | 0.020 | 0.003 | 0.075 | 0.010 | 0.009 |
| XEC | 0.122 | -0.014 | -0.006 | -0.053 | -0.059 | -0.001 | -0.056 | -0.000 | 0.050 | 0.011 | -0.000 | -0.097 | 0.027 | 0.093 | 0.001 | -0.015 | -0.005 | 1.000 | -0.061 | 0.056 | -0.043 | -0.076 | -0.056 | 0.024 |
| BEL | 0.028 | -0.145 | 0.064 | -0.054 | -0.034 | 0.077 | 0.059 | -0.014 | 0.050 | -0.002 | -0.003 | -0.006 | -0.057 | -0.088 | -0.039 | 0.059 | -0.030 | -0.061 | 1.000 | -0.022 | 0.011 | 0.028 | 0.015 | 0.009 |
| ANTHROPIC | 0.151 | -0.011 | -0.028 | 0.036 | 0.045 | 0.037 | -0.057 | -0.010 | 0.009 | -0.108 | -0.004 | 0.015 | 0.094 | 0.020 | 0.032 | 0.003 | 0.020 | 0.056 | -0.022 | 1.000 | 0.048 | 0.014 | -0.037 | -0.017 |
| BAR | 0.115 | -0.058 | 0.022 | -0.015 | 0.018 | 0.077 | -0.125 | -0.041 | -0.100 | 0.055 | -0.013 | -0.055 | 0.032 | -0.007 | 0.009 | 0.076 | 0.003 | -0.043 | 0.011 | 0.048 | 1.000 | -0.009 | 0.001 | -0.018 |
| Q | 0.026 | 0.015 | -0.066 | 0.134 | -0.002 | 0.087 | 0.094 | 0.008 | 0.036 | -0.029 | -0.019 | 0.020 | 0.094 | 0.006 | -0.049 | 0.003 | 0.075 | -0.076 | 0.028 | 0.014 | -0.009 | 1.000 | -0.014 | -0.061 |
| DEXE | 0.004 | -0.024 | 0.193 | -0.024 | 0.033 | -0.020 | 0.011 | -0.018 | 0.010 | 0.120 | 0.062 | 0.077 | -0.039 | 0.082 | 0.011 | -0.011 | 0.010 | -0.056 | 0.015 | -0.037 | 0.001 | -0.014 | 1.000 | 0.031 |
| B | -0.036 | -0.061 | -0.068 | -0.107 | 0.043 | 0.017 | 0.016 | 0.040 | -0.059 | 0.051 | -0.149 | -0.063 | 0.113 | 0.004 | -0.020 | -0.034 | 0.009 | 0.024 | 0.009 | -0.017 | -0.018 | -0.061 | 0.031 | 1.000 |

The complete 116 × 116 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| RATS | 1000RATSUSDT | 66.0% | 58.3% | BARUSDT | 0.220 |
| MBL | MBLUSDT | 66.1% | 58.2% | BTCUSDT | 0.375 |
| MYX | MYXUSDT | 66.5% | 57.9% | BTCUSDT | 0.302 |
| PIEVERSE | PIEVERSEUSDT | 66.5% | 57.9% | 币安人生USDT | 0.356 |
| DIS | DISUSDT | 66.6% | 57.7% | VUSDT | 0.317 |
| RAVE | RAVEUSDT | 67.2% | 57.3% | ACTUSDT | 0.348 |
| MAV | MAVUSDT | 67.2% | 57.3% | BTCUSDT | 0.450 |
| GPS | GPSUSDT | 67.3% | 57.2% | TLMUSDT | 0.212 |
| VANA | VANAUSDT | 67.4% | 57.1% | EGLDUSDT | 0.363 |
| HIMS | HIMSUSDT | 67.7% | 56.8% | BTCUSDT | 0.387 |
| IOTA | IOTAUSDT | 67.8% | 56.8% | BTCUSDT | 0.424 |
| BEAT | BEATUSDT | 67.8% | 56.8% | PIPPINUSDT | 0.239 |
| STG | STGUSDT | 67.8% | 56.7% | BTCUSDT | 0.322 |
| LDO | LDOUSDT | 67.9% | 56.7% | BTCUSDT | 0.355 |
| KITE | KITEUSDT | 68.1% | 56.5% | BTCUSDT | 0.331 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

