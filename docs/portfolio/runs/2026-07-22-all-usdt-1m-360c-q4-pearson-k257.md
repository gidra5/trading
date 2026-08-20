# Binance portfolio basis

Generated 2026-07-23T19:10:14.687Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-22T18:00Z through 2026-07-22T23:59Z
- Sampling: exactly 360 1m log returns (0.3 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 1.0% of the maximum unexplained variance, select the largest mean absolute 1m return
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
| 6 | ONE | ONEUSDT (spot) | spot, usdm-futures | 99.6% | 0.061 | BTCUSDT | 34.95 bp | 100.0% | 0.2 | 223.5K | 382.6% |
| 7 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 99.4% | 0.077 | BTTCUSDT | 33.19 bp | 100.0% | 0 | 4M | 327.3% |
| 8 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.4% | 0.095 | BTTCUSDT | 31.24 bp | 100.0% | 0 | 26.5M | 301.7% |
| 9 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 99.5% | 0.070 | AKEUSDT | 26.24 bp | 100.0% | 0 | 5.3M | 264.9% |
| 10 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 99.3% | 0.068 | AKEUSDT | 25.51 bp | 100.0% | 0 | 12.8M | 255.8% |
| 11 | DODO | DODOUSDT (spot) | spot | 99.3% | 0.074 | BLESSUSDT | 22.48 bp | 100.0% | 0 | 1.1M | 232.8% |
| 12 | SHAZ | SHAZUSDT (usdm-futures) | usdm-futures | 99.2% | 0.083 | ONEUSDT | 25.24 bp | 100.0% | 0.1 | 6.1M | 377.1% |
| 13 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 99.3% | 0.068 | ONEUSDT | 18.21 bp | 100.0% | 0 | 4.8M | 249.9% |
| 14 | CAT | 1000CATUSDT (spot) | spot, usdm-futures | 99.2% | 0.088 | RIFUSDT | 14.67 bp | 100.0% | 0.1 | 4K | 241.5% |
| 15 | B | BUSDT (usdm-futures) | usdm-futures | 99.0% | 0.064 | RIFUSDT | 10.92 bp | 100.0% | 0.1 | 1.8M | 127.3% |
| 16 | QUICK | QUICKUSDT (spot) | spot | 99.0% | 0.083 | XNOUSDT | 9.05 bp | 100.0% | 0.4 | 4.5K | 165.7% |
| 17 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 98.8% | 0.111 | BTTCUSDT | 21.52 bp | 100.0% | 0.2 | 1.4M | 228.5% |
| 18 | MINIMAX | MINIMAXUSDT (usdm-futures) | usdm-futures | 99.0% | 0.078 | BROCCOLIF3BUSDT | 9.03 bp | 100.0% | 0.1 | 845.7K | 129.0% |
| 19 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 98.6% | 0.115 | BTCUSDT | 12.72 bp | 100.0% | 0 | 483.5K | 147.7% |
| 20 | STRAX | STRAXUSDT (spot) | spot | 98.6% | 0.069 | QUICKUSDT | 4.52 bp | 100.0% | 0.3 | 31.7K | 63.1% |
| 21 | MMT | MMTUSDT (spot) | spot, usdm-futures | 98.6% | 0.076 | BTCUSDT | 5.75 bp | 100.0% | 0.1 | 136K | 77.1% |
| 22 | EWT | EWTUSDT (usdm-futures) | usdm-futures | 98.4% | 0.077 | BTCUSDT | 3.48 bp | 100.0% | 0.1 | 227.4K | 45.3% |
| 23 | ID | IDUSDT (spot) | spot, usdm-futures | 98.9% | 0.061 | QUICKUSDT | 2.73 bp | 100.0% | 0.7 | 42.1K | 69.8% |
| 24 | C | CUSDT (spot) | spot, usdm-futures | 98.4% | 0.082 | STRAXUSDT | 2.31 bp | 100.0% | 0.7 | 32.6K | 50.7% |
| 25 | LLY | LLYUSDT (usdm-futures) | usdm-futures | 98.8% | 0.065 | RIFUSDT | 1.58 bp | 100.0% | 0.1 | 118.3K | 29.4% |
| 26 | MET | METUSDT (spot) | spot, usdm-futures | 97.9% | 0.080 | ONEUSDT | 6.32 bp | 100.0% | 0.2 | 146.2K | 95.0% |
| 27 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 97.9% | 0.093 | ONEUSDT | 5.85 bp | 100.0% | 0.2 | 47.9K | 67.5% |
| 28 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 97.6% | 0.088 | DEXEUSDT | 11.45 bp | 100.0% | 0 | 240.2K | 119.8% |
| 29 | ATM | ATMUSDT (spot) | spot | 97.5% | 0.107 | AKEUSDT | 14.87 bp | 100.0% | 0.2 | 241.5K | 175.4% |
| 30 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 97.5% | 0.089 | ONEUSDT | 5.23 bp | 100.0% | 0.8 | 17.4K | 191.0% |
| 31 | AT | ATUSDT (spot) | spot, usdm-futures | 97.5% | 0.091 | BLESSUSDT | 2.54 bp | 100.0% | 0.3 | 20.6K | 48.6% |
| 32 | NEWT | NEWTUSDT (spot) | spot, usdm-futures | 97.6% | 0.094 | RIFUSDT | 1.77 bp | 100.0% | 1.3 | 69.2K | 53.9% |
| 33 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 97.2% | 0.124 | B2USDT | 1.28 bp | 100.0% | 0.1 | 44.4K | 25.9% |
| 34 | MTL | MTLUSDT (spot) | spot, usdm-futures | 97.5% | 0.078 | BTCUSDT | 1.11 bp | 100.0% | 1.6 | 2.7K | 56.3% |
| 35 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 96.7% | 0.139 | ONEUSDT | 17.64 bp | 100.0% | 0 | 571.6K | 169.9% |
| 36 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 96.4% | 0.108 | DEXEUSDT | 13.70 bp | 100.0% | 0.1 | 1.5M | 154.8% |
| 37 | BAN | BANUSDT (usdm-futures) | usdm-futures | 96.7% | 0.108 | XNOUSDT | 6.48 bp | 100.0% | 0.1 | 2M | 114.6% |
| 38 | REQ | REQUSDT (spot) | spot | 96.1% | 0.096 | NEWTUSDT | 4.64 bp | 100.0% | 0.5 | 8.3K | 85.4% |
| 39 | SOMI | SOMIUSDT (spot) | spot, usdm-futures | 96.1% | 0.108 | HUMAUSDT | 3.51 bp | 100.0% | 0.4 | 38.2K | 58.6% |
| 40 | TENCENT | TENCENTUSDT (usdm-futures) | usdm-futures | 96.2% | 0.095 | BTCUSDT | 3.21 bp | 100.0% | 0.1 | 90.3K | 40.3% |
| 41 | SNX | SNXUSDT (spot) | spot, usdm-futures | 95.9% | 0.135 | METUSDT | 8.73 bp | 100.0% | 0.6 | 520.8K | 152.3% |
| 42 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 95.9% | 0.099 | BUSDT | 7.28 bp | 100.0% | 0.1 | 111.2K | 76.0% |
| 43 | DGB | DGBUSDT (spot) | spot | 95.4% | 0.123 | RIFUSDT | 19.57 bp | 100.0% | 0.2 | 105.8K | 333.8% |
| 44 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 95.6% | 0.120 | BTTCUSDT | 13.40 bp | 100.0% | 0 | 3.3M | 129.0% |
| 45 | VIC | VICUSDT (spot) | spot, usdm-futures | 95.4% | 0.148 | AGTUSDT | 6.21 bp | 100.0% | 0.6 | 28.3K | 101.5% |
| 46 | AUDIO | AUDIOUSDT (spot) | spot | 95.4% | 0.103 | 1000CATUSDT | 4.59 bp | 100.0% | 0.3 | 47.7K | 59.6% |
| 47 | TURTLE | TURTLEUSDT (spot) | spot, usdm-futures | 95.7% | 0.117 | MIRAUSDT | 3.52 bp | 100.0% | 0.7 | 41.3K | 77.5% |
| 48 | RIVN | RIVNUSDT (usdm-futures) | usdm-futures | 95.4% | 0.139 | B2USDT | 3.48 bp | 100.0% | 0.1 | 164.6K | 59.1% |
| 49 | IQ | IQUSDT (spot) | spot | 95.3% | 0.101 | SPORTFUNUSDT | 2.08 bp | 100.0% | 1.3 | 6.7K | 51.8% |
| 50 | AWE | AWEUSDT (spot) | spot, usdm-futures | 94.9% | 0.124 | SHAZUSDT | 1.38 bp | 100.0% | 0.9 | 19.1K | 43.7% |
| 51 | ACX | ACXUSDT (spot) | spot, usdm-futures | 95.2% | 0.129 | COOKIEUSDT | 0.92 bp | 100.0% | 0.6 | 14.7K | 21.6% |
| 52 | SPK | SPKUSDT (spot) | spot, usdm-futures | 94.5% | 0.121 | RECALLUSDT | 2.43 bp | 100.0% | 0.4 | 116.2K | 53.9% |
| 53 | TAC | TACUSDT (usdm-futures) | usdm-futures | 94.0% | 0.097 | NEWTUSDT | 14.20 bp | 100.0% | 0 | 1.2M | 151.6% |
| 54 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 94.0% | 0.150 | MINIMAXUSDT | 3.18 bp | 100.0% | 0.7 | 30.5K | 78.8% |
| 55 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 93.7% | 0.160 | BTCUSDT | 7.41 bp | 100.0% | 0.1 | 617.1K | 78.3% |
| 56 | WEN | WENUSDT (usdm-futures) | usdm-futures | 93.7% | 0.129 | XNOUSDT | 3.78 bp | 100.0% | 0.1 | 120K | 65.3% |
| 57 | ASTR | ASTRUSDT (spot) | spot, usdm-futures | 93.7% | 0.124 | AKEUSDT | 2.85 bp | 100.0% | 0.4 | 16.1K | 50.8% |
| 58 | AIN | AINUSDT (usdm-futures) | usdm-futures | 93.3% | 0.127 | SHAZUSDT | 10.51 bp | 100.0% | 0 | 112.9K | 102.9% |
| 59 | XVG | XVGUSDT (spot) | spot, usdm-futures | 93.5% | 0.099 | TURTLEUSDT | 2.83 bp | 100.0% | 0.7 | 18.2K | 59.6% |
| 60 | AVGO | AVGOBUSDT (spot) | spot, usdm-futures | 93.0% | 0.139 | RECALLUSDT | 4.16 bp | 100.0% | 0 | 15.4K | 82.9% |
| 61 | ZKC | ZKCUSDT (spot) | spot, usdm-futures | 93.1% | 0.107 | XNOUSDT | 2.27 bp | 100.0% | 0.6 | 59.8K | 54.5% |
| 62 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 92.6% | 0.140 | METUSDT | 3.09 bp | 100.0% | 0 | 88.5K | 44.4% |
| 63 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 92.8% | 0.130 | EWTUSDT | 2.26 bp | 100.0% | 0.9 | 14.3K | 59.9% |
| 64 | RLC | RLCUSDT (spot) | spot, usdm-futures | 92.7% | 0.123 | BUSDT | 1.25 bp | 100.0% | 0.6 | 3.9K | 39.5% |
| 65 | V | VUSDT (usdm-futures) | usdm-futures | 92.5% | 0.113 | RIFUSDT | 1.23 bp | 100.0% | 0 | 54.6K | 21.8% |
| 66 | URNM | URNMUSDT (usdm-futures) | usdm-futures | 92.4% | 0.109 | RECALLUSDT | 1.29 bp | 100.0% | 0.1 | 45K | 25.9% |
| 67 | BNC | BNCUSDT (usdm-futures) | usdm-futures | 92.1% | 0.129 | TENCENTUSDT | 5.50 bp | 100.0% | 0.2 | 80.9K | 73.6% |
| 68 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 92.1% | 0.122 | BANUSDT | 1.22 bp | 100.0% | 0 | 115.8K | 17.0% |
| 69 | SUN | SUNUSDT (spot) | spot, usdm-futures | 92.4% | 0.128 | ATUSDT | 0.66 bp | 100.0% | 0.8 | 39.4K | 14.3% |
| 70 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 91.4% | 0.116 | COOKIEUSDT | 16.30 bp | 100.0% | 0.2 | 985.9K | 227.3% |
| 71 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 91.4% | 0.132 | ONEUSDT | 10.62 bp | 100.0% | 0 | 9.1M | 185.1% |
| 72 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 90.9% | 0.146 | BTCUSDT | 6.07 bp | 100.0% | 0.1 | 172.6K | 67.1% |
| 73 | BAR | BARUSDT (spot) | spot | 90.7% | 0.132 | REQUSDT | 7.22 bp | 100.0% | 0.4 | 69.7K | 118.0% |
| 74 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 91.0% | 0.166 | DEXEUSDT | 5.74 bp | 100.0% | 0.1 | 503K | 86.5% |
| 75 | QCOM | QCOMBUSDT (spot) | spot, usdm-futures | 90.5% | 0.132 | EWTUSDT | 3.64 bp | 100.0% | 0 | 13.1K | 69.8% |
| 76 | CHR | CHRUSDT (spot) | spot, usdm-futures | 90.6% | 0.115 | AKEUSDT | 2.57 bp | 100.0% | 0.5 | 16.3K | 63.1% |
| 77 | CYBER | CYBERUSDT (spot) | spot, usdm-futures | 90.8% | 0.110 | MIRAUSDT | 1.65 bp | 100.0% | 1.7 | 34.1K | 55.5% |
| 78 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 90.0% | 0.143 | IQUSDT | 17.12 bp | 100.0% | 0.1 | 3.8M | 179.4% |
| 79 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 89.6% | 0.170 | BTCUSDT | 13.60 bp | 100.0% | 0 | 1.2M | 171.5% |
| 80 | IBM | IBMBUSDT (spot) | spot, usdm-futures | 89.7% | 0.157 | MINIMAXUSDT | 11.56 bp | 100.0% | 0.1 | 666.6K | 189.8% |
| 81 | VANA | VANAUSDT (spot) | spot, usdm-futures | 89.8% | 0.149 | BTTCUSDT | 6.66 bp | 100.0% | 0.1 | 1.7M | 71.0% |
| 82 | M | MUSDT (usdm-futures) | usdm-futures | 89.9% | 0.135 | XNOUSDT | 6.22 bp | 100.0% | 0.1 | 517K | 64.1% |
| 83 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 89.6% | 0.153 | METUSDT | 2.27 bp | 100.0% | 0.2 | 83.4K | 34.0% |
| 84 | KNC | KNCUSDT (spot) | spot, usdm-futures | 89.9% | 0.114 | XVGUSDT | 1.32 bp | 100.0% | 0.8 | 18K | 37.8% |
| 85 | TA | TAUSDT (usdm-futures) | usdm-futures | 88.9% | 0.149 | BTCUSDT | 7.84 bp | 100.0% | 0.1 | 455.1K | 81.6% |
| 86 | HPE | HPEUSDT (usdm-futures) | usdm-futures | 88.7% | 0.133 | EWTUSDT | 6.02 bp | 100.0% | 0 | 225K | 80.7% |
| 87 | XVS | XVSUSDT (spot) | spot, usdm-futures | 88.9% | 0.148 | BTCUSDT | 3.10 bp | 100.0% | 1.1 | 11.2K | 86.0% |
| 88 | YFI | YFIUSDT (spot) | spot, usdm-futures | 88.7% | 0.169 | BTCUSDT | 2.72 bp | 100.0% | 0.4 | 74.4K | 54.5% |
| 89 | WAXP | WAXPUSDT (spot) | spot, usdm-futures | 88.6% | 0.162 | CUSDT | 2.02 bp | 100.0% | 0.6 | 6.4K | 55.2% |
| 90 | FF | FFUSDT (spot) | spot, usdm-futures | 88.1% | 0.132 | ACXUSDT | 2.01 bp | 100.0% | 0.3 | 108.4K | 26.4% |
| 91 | POLYX | POLYXUSDT (spot) | spot, usdm-futures | 88.3% | 0.152 | BTCUSDT | 1.83 bp | 100.0% | 0.7 | 13.7K | 56.2% |
| 92 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 87.4% | 0.156 | EVAAUSDT | 15.59 bp | 100.0% | 0 | 4.4M | 168.6% |
| 93 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 87.5% | 0.132 | KNCUSDT | 14.53 bp | 100.0% | 0 | 962.6K | 140.2% |
| 94 | ZHIPU | ZHIPUUSDT (usdm-futures) | usdm-futures | 87.2% | 0.175 | IBMBUSDT | 11.67 bp | 100.0% | 0 | 4.4M | 128.0% |
| 95 | FWDI | FWDIUSDT (usdm-futures) | usdm-futures | 87.2% | 0.143 | BTCUSDT | 6.84 bp | 100.0% | 0.1 | 141.3K | 87.5% |
| 96 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 87.4% | 0.123 | MMTUSDT | 3.16 bp | 100.0% | 0.1 | 416.7K | 42.6% |
| 97 | OPN | OPNUSDT (spot) | spot, usdm-futures | 86.7% | 0.197 | URNMUSDT | 14.52 bp | 100.0% | 0.1 | 2.4M | 161.0% |
| 98 | HEMI | HEMIUSDT (spot) | spot, usdm-futures | 86.8% | 0.192 | BTCUSDT | 11.28 bp | 100.0% | 0.2 | 359.6K | 133.1% |
| 99 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 86.3% | 0.224 | COOKIEUSDT | 11.22 bp | 100.0% | 0.1 | 639K | 210.8% |
| 100 | XEC | XECUSDT (spot) | spot, usdm-futures | 86.4% | 0.183 | SPORTFUNUSDT | 9.55 bp | 100.0% | 0.1 | 278.8K | 113.3% |
| 101 | RE | REUSDT (spot) | spot, usdm-futures | 86.0% | 0.147 | ATHUSDT | 17.39 bp | 100.0% | 0 | 6.5M | 167.5% |
| 102 | NOK | NOKBUSDT (spot) | spot, usdm-futures | 86.3% | 0.123 | QUICKUSDT | 8.31 bp | 100.0% | 0.1 | 149.3K | 135.7% |
| 103 | BANANA | BANANAUSDT (spot) | spot, usdm-futures | 85.9% | 0.140 | MINIMAXUSDT | 8.22 bp | 100.0% | 0.1 | 157K | 98.2% |
| 104 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 85.7% | 0.132 | SUNUSDT | 8.12 bp | 100.0% | 0.1 | 377.6K | 84.6% |
| 105 | TFUEL | TFUELUSDT (spot) | spot | 85.5% | 0.143 | EWTUSDT | 6.81 bp | 100.0% | 0.2 | 55.4K | 104.6% |
| 106 | GMX | GMXUSDT (spot) | spot, usdm-futures | 85.2% | 0.154 | ATHUSDT | 5.65 bp | 100.0% | 1.2 | 41.9K | 77.4% |
| 107 | JOE | JOEUSDT (spot) | spot, usdm-futures | 85.4% | 0.145 | AIAUSDT | 4.99 bp | 100.0% | 0.6 | 30.4K | 108.4% |
| 108 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 85.2% | 0.140 | BTCUSDT | 4.85 bp | 100.0% | 0.4 | 109.1K | 65.5% |
| 109 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 85.0% | 0.131 | BSBUSDT | 4.65 bp | 100.0% | 0.4 | 96.3K | 82.5% |
| 110 | ZK | ZKUSDT (spot) | spot, usdm-futures | 84.7% | 0.137 | IDOLUSDT | 5.14 bp | 100.0% | 0.1 | 149.3K | 68.0% |
| 111 | AXTI | AXTIBUSDT (spot) | spot, usdm-futures | 84.6% | 0.177 | BANUSDT | 14.79 bp | 100.0% | 0 | 31.5K | 303.5% |
| 112 | XAN | XANUSDT (usdm-futures) | usdm-futures | 84.2% | 0.148 | QUICKUSDT | 9.09 bp | 100.0% | 0 | 303.2K | 112.4% |
| 113 | HK1810 | HK1810USDT (usdm-futures) | usdm-futures | 84.2% | 0.126 | KNCUSDT | 4.01 bp | 100.0% | 0.1 | 119.4K | 54.8% |
| 114 | SCRT | SCRTUSDT (spot) | spot, usdm-futures | 84.3% | 0.145 | BARUSDT | 3.21 bp | 100.0% | 0.5 | 22.8K | 66.4% |
| 115 | XAG | XAGUSDT (usdm-futures) | usdm-futures | 84.2% | 0.196 | BTCUSDT | 3.12 bp | 100.0% | 0.1 | 143.3M | 31.9% |
| 116 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 83.9% | 0.122 | IDUSDT | 2.43 bp | 100.0% | 0.1 | 2.9M | 31.8% |
| 117 | HOT | HOTUSDT (spot) | spot, usdm-futures | 84.2% | 0.136 | ASTRUSDT | 2.35 bp | 100.0% | 0.6 | 9.6K | 64.1% |
| 118 | META | METABUSDT (spot) | spot, usdm-futures | 84.2% | 0.161 | BNCUSDT | 2.23 bp | 100.0% | 0 | 16.1K | 45.5% |
| 119 | BANK | BANKUSDT (spot) | spot, usdm-futures | 83.1% | 0.154 | SPKUSDT | 46.41 bp | 100.0% | 0 | 18.5M | 496.0% |
| 120 | ORCL | ORCLBUSDT (spot) | spot, usdm-futures | 83.2% | 0.156 | BTTCUSDT | 3.31 bp | 100.0% | 0.1 | 32.5K | 78.1% |
| 121 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 83.2% | 0.144 | ATHUSDT | 2.82 bp | 100.0% | 0.7 | 74.9K | 70.7% |
| 122 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 82.8% | 0.158 | AUCTIONUSDT | 4.09 bp | 100.0% | 0.1 | 209K | 49.9% |
| 123 | BREV | BREVUSDT (spot) | spot, usdm-futures | 82.9% | 0.173 | BTCUSDT | 2.34 bp | 100.0% | 0.8 | 15.7K | 49.8% |
| 124 | CITY | CITYUSDT (spot) | spot | 82.4% | 0.148 | BULLAUSDT | 9.14 bp | 100.0% | 0.3 | 100.3K | 111.3% |
| 125 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 82.3% | 0.189 | ATUSDT | 7.37 bp | 100.0% | 0.2 | 319.1K | 93.4% |
| 126 | USTC | USTCUSDT (spot) | spot, usdm-futures | 82.2% | 0.150 | KNCUSDT | 2.98 bp | 100.0% | 0.8 | 24.9K | 68.6% |
| 127 | DOLO | DOLOUSDT (spot) | spot, usdm-futures | 82.3% | 0.199 | BLUAIUSDT | 1.86 bp | 100.0% | 0.4 | 11.6K | 40.1% |
| 128 | METIS | METISUSDT (spot) | spot, usdm-futures | 82.4% | 0.157 | WAXPUSDT | 1.10 bp | 100.0% | 1.8 | 9.2K | 45.5% |
| 129 | USELESS | USELESSUSDT (usdm-futures) | usdm-futures | 81.4% | 0.206 | BTCUSDT | 10.00 bp | 100.0% | 0 | 1.3M | 144.3% |
| 130 | APR | APRUSDT (usdm-futures) | usdm-futures | 81.4% | 0.140 | SOMIUSDT | 8.45 bp | 100.0% | 0.1 | 532.6K | 90.8% |
| 131 | OPG | OPGUSDT (spot) | spot, usdm-futures | 81.6% | 0.239 | BTCUSDT | 4.84 bp | 100.0% | 0.2 | 133.5K | 67.4% |
| 132 | SC | SCUSDT (spot) | spot | 81.5% | 0.154 | AIAUSDT | 3.02 bp | 100.0% | 0.5 | 10.5K | 58.1% |
| 133 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 80.7% | 0.179 | BTCUSDT | 5.80 bp | 100.0% | 0.1 | 208.8K | 73.5% |
| 134 | ESP | ESPUSDT (spot) | spot, usdm-futures | 80.7% | 0.135 | CRWDUSDT | 4.76 bp | 100.0% | 0.1 | 66.9K | 51.7% |
| 135 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 80.5% | 0.162 | VANAUSDT | 2.81 bp | 100.0% | 0.2 | 127.8K | 34.0% |
| 136 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 80.8% | 0.182 | GWEIUSDT | 2.31 bp | 100.0% | 0 | 61.3K | 41.4% |
| 137 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 80.4% | 0.128 | JASMYUSDT | 8.89 bp | 100.0% | 0.1 | 586.3K | 101.5% |
| 138 | AI | AIUSDT (spot) | spot | 79.7% | 0.154 | ZHIPUUSDT | 5.81 bp | 100.0% | 0.9 | 28.3K | 126.4% |
| 139 | ERA | ERAUSDT (spot) | spot, usdm-futures | 79.5% | 0.144 | BNCUSDT | 30.19 bp | 100.0% | 0.1 | 1.9M | 329.2% |
| 140 | KLAC | KLACUSDT (usdm-futures) | usdm-futures | 79.3% | 0.159 | EVAAUSDT | 6.13 bp | 100.0% | 0 | 176.7K | 82.4% |
| 141 | CFX | CFXUSDT (spot) | spot, usdm-futures | 79.6% | 0.221 | BTCUSDT | 5.34 bp | 100.0% | 0.1 | 164.2K | 70.8% |
| 142 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 78.7% | 0.167 | GWEIUSDT | 15.77 bp | 100.0% | 0.1 | 695.3K | 182.0% |
| 143 | 4 | 4USDT (usdm-futures) | usdm-futures | 78.6% | 0.178 | CRWDUSDT | 7.25 bp | 100.0% | 0 | 156.4K | 82.3% |
| 144 | ACH | ACHUSDT (spot) | spot, usdm-futures | 78.7% | 0.172 | BTCUSDT | 5.50 bp | 100.0% | 0.4 | 66.8K | 85.0% |
| 145 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 78.2% | 0.150 | IBMBUSDT | 5.83 bp | 100.0% | 0.1 | 109.1K | 75.4% |
| 146 | MEME | MEMEUSDT (spot) | spot, usdm-futures | 78.6% | 0.144 | OPGUSDT | 3.82 bp | 100.0% | 0.4 | 30.4K | 65.6% |
| 147 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 77.7% | 0.144 | TENCENTUSDT | 17.64 bp | 100.0% | 0 | 1.5M | 203.8% |
| 148 | ENS | ENSUSDT (spot) | spot, usdm-futures | 77.5% | 0.201 | BTCUSDT | 6.87 bp | 100.0% | 0.3 | 168.6K | 92.0% |
| 149 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 77.4% | 0.202 | BTCUSDT | 10.74 bp | 100.0% | 0 | 1.3M | 105.6% |
| 150 | TST | TSTUSDT (spot) | spot, usdm-futures | 77.5% | 0.138 | ATMUSDT | 5.14 bp | 100.0% | 0.3 | 76.7K | 98.4% |
| 151 | STG | STGUSDT (spot) | spot, usdm-futures | 77.5% | 0.157 | CYBERUSDT | 2.31 bp | 100.0% | 0.4 | 34.7K | 45.8% |
| 152 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 76.7% | 0.210 | BTCUSDT | 14.23 bp | 100.0% | 0 | 1.1M | 150.1% |
| 153 | SOPH | SOPHUSDT (spot) | spot, usdm-futures | 76.5% | 0.181 | SHAZUSDT | 3.69 bp | 100.0% | 0.4 | 14.3K | 69.6% |
| 154 | SNOW | SNOWUSDT (usdm-futures) | usdm-futures | 76.6% | 0.175 | CRWDUSDT | 3.59 bp | 100.0% | 0 | 42.3K | 58.7% |
| 155 | COST | COSTUSDT (usdm-futures) | usdm-futures | 76.8% | 0.154 | GMXUSDT | 1.05 bp | 100.0% | 0.1 | 90.2K | 21.6% |
| 156 | PROM | PROMUSDT (spot) | spot, usdm-futures | 76.1% | 0.154 | XNOUSDT | 14.97 bp | 100.0% | 0.1 | 153K | 155.2% |
| 157 | GLMR | GLMRUSDT (spot) | spot | 76.0% | 0.136 | WAXPUSDT | 8.10 bp | 100.0% | 0.1 | 139.2K | 103.6% |
| 158 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 75.5% | 0.163 | XNOUSDT | 10.37 bp | 100.0% | 0.1 | 222.7K | 114.6% |
| 159 | T | TUSDT (spot) | spot, usdm-futures | 75.4% | 0.187 | SPELLUSDT | 6.47 bp | 100.0% | 0.4 | 144.6K | 99.4% |
| 160 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 75.5% | 0.138 | NEWTUSDT | 6.11 bp | 100.0% | 0.1 | 63.6K | 96.9% |
| 161 | OPEN | OPENUSDT (spot) | spot, usdm-futures | 75.2% | 0.197 | BTCUSDT | 5.07 bp | 100.0% | 0.3 | 80.7K | 67.8% |
| 162 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 74.6% | 0.145 | HUMAUSDT | 13.52 bp | 100.0% | 0 | 499.2K | 153.5% |
| 163 | CC | CCUSDT (usdm-futures) | usdm-futures | 74.5% | 0.178 | BTCUSDT | 3.33 bp | 100.0% | 0 | 555.8K | 37.9% |
| 164 | KSM | KSMUSDT (spot) | spot, usdm-futures | 74.4% | 0.185 | HANAUSDT | 4.42 bp | 100.0% | 0.4 | 23.6K | 86.9% |
| 165 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 74.3% | 0.166 | SPKUSDT | 8.77 bp | 100.0% | 0.1 | 156.1K | 87.9% |
| 166 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 74.0% | 0.175 | HYUNDAIUSDT | 7.83 bp | 100.0% | 0.1 | 795.5K | 89.4% |
| 167 | KITE | KITEUSDT (spot) | spot, usdm-futures | 74.2% | 0.150 | FWDIUSDT | 7.50 bp | 100.0% | 0.2 | 8.6M | 76.5% |
| 168 | CKB | CKBUSDT (spot) | spot, usdm-futures | 74.2% | 0.227 | POLYXUSDT | 2.83 bp | 100.0% | 0.3 | 29.8K | 54.8% |
| 169 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 73.3% | 0.157 | XECUSDT | 11.22 bp | 100.0% | 0.1 | 369.5K | 120.3% |
| 170 | BE | BEUSDT (usdm-futures) | usdm-futures | 73.0% | 0.191 | AWEUSDT | 9.82 bp | 100.0% | 0 | 2.2M | 105.6% |
| 171 | IOTA | IOTAUSDT (spot) | spot, usdm-futures | 72.9% | 0.205 | QUICKUSDT | 7.46 bp | 100.0% | 0.4 | 136.5K | 110.9% |
| 172 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 72.8% | 0.150 | BSBUSDT | 6.06 bp | 100.0% | 0.3 | 156.1K | 78.1% |
| 173 | WET | WETUSDT (usdm-futures) | usdm-futures | 72.7% | 0.142 | 1000CATUSDT | 5.18 bp | 100.0% | 0.1 | 199K | 62.6% |
| 174 | ZM | ZMUSDT (usdm-futures) | usdm-futures | 72.9% | 0.174 | BSBUSDT | 3.46 bp | 100.0% | 0 | 70.1K | 53.0% |
| 175 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 72.2% | 0.180 | KSTRUSDT | 17.53 bp | 100.0% | 0 | 982.2K | 176.0% |
| 176 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 72.1% | 0.161 | SPORTFUNUSDT | 10.32 bp | 100.0% | 0.1 | 463.2K | 109.7% |
| 177 | WIN | WINUSDT (spot) | spot | 71.7% | 0.138 | BRKBUSDT | 10.90 bp | 100.0% | 0.1 | 24.7K | 107.9% |
| 178 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 71.5% | 0.135 | REUSDT | 15.44 bp | 100.0% | 0 | 456.2K | 155.2% |
| 179 | RKLB | RKLBBUSDT (spot) | spot, usdm-futures | 71.5% | 0.169 | AVGOBUSDT | 8.76 bp | 100.0% | 0.1 | 48.3K | 128.8% |
| 180 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 71.5% | 0.235 | BTCUSDT | 7.91 bp | 100.0% | 0 | 586.2K | 78.3% |
| 181 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 71.7% | 0.183 | CKBUSDT | 2.79 bp | 100.0% | 0.4 | 12.3K | 53.1% |
| 182 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 71.2% | 0.191 | BTCUSDT | 5.16 bp | 100.0% | 0.2 | 75.4K | 68.7% |
| 183 | BROCCOLI714 | BROCCOLI714USDT (spot) | spot, usdm-futures | 71.0% | 0.176 | USTCUSDT | 2.94 bp | 100.0% | 0.4 | 34.7K | 53.6% |
| 184 | ICX | ICXUSDT (spot) | spot, usdm-futures | 71.2% | 0.212 | XVSUSDT | 2.13 bp | 100.0% | 1.3 | 12.9K | 86.3% |
| 185 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 70.5% | 0.180 | FWDIUSDT | 7.82 bp | 100.0% | 0 | 383.8K | 80.7% |
| 186 | UMA | UMAUSDT (spot) | spot, usdm-futures | 70.8% | 0.191 | ASTRUSDT | 1.13 bp | 100.0% | 0.8 | 6.4K | 40.1% |
| 187 | ACE | ACEUSDT (spot) | spot, usdm-futures | 70.1% | 0.159 | ACHUSDT | 14.90 bp | 100.0% | 0.1 | 386.8K | 155.8% |
| 188 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 70.0% | 0.219 | HANAUSDT | 2.49 bp | 100.0% | 0 | 186K | 42.4% |
| 189 | TURBO | TURBOUSDT (spot) | spot, usdm-futures | 69.5% | 0.168 | KSMUSDT | 2.44 bp | 100.0% | 0.4 | 20.7K | 45.8% |
| 190 | SYN | SYNUSDT (spot) | spot, usdm-futures | 69.3% | 0.134 | ATHUSDT | 10.81 bp | 100.0% | 0 | 551.1K | 104.9% |
| 191 | 我踏马来了 | 我踏马来了USDT (usdm-futures) | usdm-futures | 69.0% | 0.187 | BTCUSDT | 5.42 bp | 100.0% | 0.1 | 218.4K | 60.0% |
| 192 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 68.9% | 0.150 | COSTUSDT | 5.26 bp | 100.0% | 0 | 167.4K | 60.9% |
| 193 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 69.2% | 0.226 | BTCUSDT | 3.24 bp | 100.0% | 0.2 | 49.2K | 42.6% |
| 194 | NMR | NMRUSDT (spot) | spot, usdm-futures | 69.1% | 0.241 | BANUSDT | 1.12 bp | 100.0% | 0.8 | 6.5K | 32.5% |
| 195 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 68.3% | 0.194 | BTCUSDT | 7.64 bp | 100.0% | 0.1 | 346.4K | 83.7% |
| 196 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 67.7% | 0.254 | ACXUSDT | 5.41 bp | 100.0% | 0.3 | 50.9K | 84.7% |
| 197 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 67.6% | 0.215 | BTCUSDT | 8.02 bp | 100.0% | 0.1 | 97.6K | 90.4% |
| 198 | GPS | GPSUSDT (spot) | spot, usdm-futures | 67.7% | 0.155 | TAGUSDT | 5.14 bp | 100.0% | 0.2 | 49.9K | 70.9% |
| 199 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 67.3% | 0.230 | METUSDT | 8.94 bp | 100.0% | 0.1 | 1.2M | 126.1% |
| 200 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 67.4% | 0.172 | XVSUSDT | 4.52 bp | 100.0% | 0.3 | 64.2K | 75.4% |
| 201 | ASML | ASMLUSDT (usdm-futures) | usdm-futures | 67.3% | 0.249 | KLACUSDT | 4.03 bp | 100.0% | 0 | 603.4K | 46.2% |
| 202 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 67.5% | 0.162 | AKEUSDT | 3.58 bp | 100.0% | 0.6 | 37K | 76.9% |
| 203 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 66.6% | 0.136 | HEMIUSDT | 8.30 bp | 100.0% | 0.1 | 256.6K | 93.2% |
| 204 | HOLO | HOLOUSDT (spot) | spot, usdm-futures | 66.6% | 0.187 | AKEUSDT | 4.01 bp | 100.0% | 0.3 | 74K | 62.0% |
| 205 | NXPC | NXPCUSDT (spot) | spot, usdm-futures | 66.4% | 0.184 | SCUSDT | 8.82 bp | 100.0% | 0.1 | 228.8K | 104.0% |
| 206 | IOST | IOSTUSDT (spot) | spot, usdm-futures | 66.3% | 0.179 | SPELLUSDT | 4.08 bp | 100.0% | 0.4 | 10.5K | 61.3% |
| 207 | LSK | LSKUSDT (spot) | spot, usdm-futures | 66.6% | 0.167 | ARPAUSDT | 2.42 bp | 100.0% | 1.1 | 8.4K | 53.2% |
| 208 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 65.7% | 0.171 | KLACUSDT | 10.38 bp | 100.0% | 0 | 246.9K | 101.3% |
| 209 | BABA | BABABUSDT (spot) | spot, usdm-futures | 65.4% | 0.173 | QCOMBUSDT | 2.07 bp | 100.0% | 0 | 8.6K | 42.1% |
| 210 | BX | BXUSDT (usdm-futures) | usdm-futures | 64.6% | 0.165 | ENSUSDT | 3.13 bp | 100.0% | 0 | 581K | 40.4% |
| 211 | VTHO | VTHOUSDT (spot) | spot, usdm-futures | 64.8% | 0.185 | BLUAIUSDT | 2.49 bp | 100.0% | 0.7 | 40.1K | 63.2% |
| 212 | STXX | STXXUSDT (usdm-futures) | usdm-futures | 64.5% | 0.215 | KLACUSDT | 8.03 bp | 100.0% | 0.1 | 427K | 100.3% |
| 213 | MOCA | MOCAUSDT (usdm-futures) | usdm-futures | 64.3% | 0.165 | BANANAUSDT | 4.93 bp | 100.0% | 0.3 | 35.6K | 59.1% |
| 214 | THETA | THETAUSDT (spot) | spot, usdm-futures | 64.1% | 0.181 | 我踏马来了USDT | 3.27 bp | 100.0% | 0.2 | 28.2K | 45.7% |
| 215 | STRC | STRCUSDT (usdm-futures) | usdm-futures | 64.4% | 0.225 | ACUUSDT | 1.00 bp | 100.0% | 0.1 | 274.5K | 18.9% |
| 216 | PYR | PYRUSDT (spot) | spot | 63.3% | 0.170 | XVGUSDT | 39.29 bp | 100.0% | 1 | 184.6K | 495.6% |
| 217 | MVLL | MVLLBUSDT (spot) | spot, usdm-futures | 63.3% | 0.168 | QCOMBUSDT | 13.00 bp | 100.0% | 0 | 14K | 280.8% |
| 218 | MOVE | MOVEUSDT (spot) | spot, usdm-futures | 63.5% | 0.189 | LLYUSDT | 2.35 bp | 100.0% | 3 | 75.1K | 107.8% |
| 219 | FTT | FTTUSDT (spot) | spot | 62.9% | 0.183 | AUDIOUSDT | 12.51 bp | 100.0% | 0.1 | 35.7K | 161.8% |
| 220 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 62.8% | 0.210 | HEMIUSDT | 8.45 bp | 100.0% | 0.1 | 454.8K | 93.5% |
| 221 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 62.8% | 0.191 | STRAXUSDT | 5.14 bp | 100.0% | 0.1 | 179.6K | 55.3% |
| 222 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 62.2% | 0.346 | METUSDT | 6.30 bp | 100.0% | 0.1 | 174.1K | 103.9% |
| 223 | AEVO | AEVOUSDT (spot) | spot, usdm-futures | 62.1% | 0.229 | BILLUSDT | 3.12 bp | 100.0% | 0.5 | 24.6K | 56.3% |
| 224 | ONDS | ONDSUSDT (usdm-futures) | usdm-futures | 61.6% | 0.244 | USELESSUSDT | 10.82 bp | 100.0% | 0.2 | 459.3K | 130.6% |
| 225 | ADX | ADXUSDT (spot) | spot | 61.3% | 0.168 | BTTCUSDT | 5.26 bp | 100.0% | 0.3 | 54K | 78.4% |
| 226 | NOT | NOTUSDT (spot) | spot, usdm-futures | 61.5% | 0.171 | VTHOUSDT | 3.43 bp | 100.0% | 0.8 | 28.9K | 70.3% |
| 227 | CATI | CATIUSDT (spot) | spot, usdm-futures | 60.8% | 0.165 | ICXUSDT | 1.99 bp | 100.0% | 0.2 | 12.5K | 32.2% |
| 228 | OG | OGUSDT (spot) | spot, usdm-futures | 60.9% | 0.187 | TAIKOUSDT | 1.64 bp | 100.0% | 0.4 | 21.3K | 31.3% |
| 229 | TUT | TUTUSDT (spot) | spot, usdm-futures | 60.4% | 0.193 | FFUSDT | 6.19 bp | 100.0% | 0.2 | 93.1K | 85.0% |
| 230 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 59.8% | 0.160 | DEXEUSDT | 8.84 bp | 100.0% | 0.1 | 81.9K | 89.5% |
| 231 | QI | QIUSDT (spot) | spot | 59.9% | 0.173 | THETAUSDT | 6.93 bp | 100.0% | 0.2 | 44.2K | 90.2% |
| 232 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 59.9% | 0.177 | SOONUSDT | 4.91 bp | 100.0% | 0.2 | 71.6K | 72.8% |
| 233 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 59.2% | 0.178 | COOKIEUSDT | 4.36 bp | 100.0% | 0.2 | 96.5K | 60.9% |
| 234 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 59.0% | 0.193 | GENIUSUSDT | 6.31 bp | 100.0% | 0.1 | 1.1M | 95.2% |
| 235 | ZIL | ZILUSDT (spot) | spot, usdm-futures | 59.1% | 0.185 | TURBOUSDT | 4.36 bp | 100.0% | 0.3 | 69.7K | 72.8% |
| 236 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 58.4% | 0.245 | HANAUSDT | 13.32 bp | 100.0% | 0.1 | 2.5M | 207.4% |
| 237 | MITO | MITOUSDT (spot) | spot, usdm-futures | 58.0% | 0.160 | 币安人生USDT | 10.64 bp | 100.0% | 0.1 | 246.3K | 121.7% |
| 238 | SHELL | SHELLUSDT (spot) | spot, usdm-futures | 57.9% | 0.164 | SPKUSDT | 5.41 bp | 100.0% | 0.8 | 57.2K | 116.3% |
| 239 | FORM | FORMUSDT (spot) | spot, usdm-futures | 57.9% | 0.206 | THETAUSDT | 2.37 bp | 100.0% | 0.5 | 38.2K | 40.9% |
| 240 | COMP | COMPUSDT (spot) | spot, usdm-futures | 57.9% | 0.181 | AEVOUSDT | 2.07 bp | 100.0% | 0.7 | 28.5K | 43.2% |
| 241 | SCR | SCRUSDT (spot) | spot, usdm-futures | 57.2% | 0.232 | YFIUSDT | 3.86 bp | 100.0% | 0.3 | 27.1K | 58.1% |
| 242 | F | FUSDT (spot) | spot, usdm-futures | 57.4% | 0.249 | ICXUSDT | 2.63 bp | 100.0% | 0.7 | 17.3K | 71.2% |
| 243 | POWR | POWRUSDT (spot) | spot, usdm-futures | 56.8% | 0.169 | NOTUSDT | 3.04 bp | 100.0% | 0.4 | 13.3K | 61.7% |
| 244 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 56.3% | 0.211 | BTCUSDT | 7.61 bp | 100.0% | 0.1 | 359.5K | 80.7% |
| 245 | QKC | QKCUSDT (spot) | spot | 56.1% | 0.200 | FWDIUSDT | 5.89 bp | 100.0% | 0.2 | 7.1K | 87.1% |
| 246 | BAT | BATUSDT (spot) | spot, usdm-futures | 56.2% | 0.186 | DOLOUSDT | 1.53 bp | 100.0% | 1.1 | 39.2K | 38.2% |
| 247 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 55.3% | 0.192 | METUSDT | 11.16 bp | 100.0% | 0.2 | 638.5K | 140.5% |
| 248 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 55.3% | 0.164 | ENSOUSDT | 9.94 bp | 100.0% | 0 | 897.7K | 97.0% |
| 249 | O | OUSDT (usdm-futures) | usdm-futures | 55.1% | 0.262 | MTLUSDT | 8.71 bp | 100.0% | 0 | 871.8K | 99.4% |
| 250 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 55.0% | 0.177 | URNMUSDT | 4.73 bp | 100.0% | 0 | 5.4M | 53.9% |
| 251 | LQTY | LQTYUSDT (spot) | spot, usdm-futures | 55.1% | 0.191 | BABABUSDT | 1.83 bp | 100.0% | 0.5 | 4.9K | 45.5% |
| 252 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 55.1% | 0.202 | NEWTUSDT | 0.89 bp | 100.0% | 0.1 | 55.9K | 24.0% |
| 253 | CIEN | CIENUSDT (usdm-futures) | usdm-futures | 54.5% | 0.161 | BEUSDT | 4.05 bp | 100.0% | 0 | 54.8K | 76.1% |
| 254 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 54.3% | 0.200 | BTCUSDT | 4.06 bp | 100.0% | 0.2 | 52.2K | 54.1% |
| 255 | EDU | EDUUSDT (spot) | spot, usdm-futures | 54.4% | 0.188 | ATHUSDT | 1.90 bp | 100.0% | 1.4 | 37.8K | 57.1% |
| 256 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 54.2% | 0.145 | ONEUSDT | 1.85 bp | 100.0% | 0.1 | 54.7K | 32.7% |
| 257 | KGST | KGSTUSDT (spot) | spot | 54.4% | 0.179 | POWRUSDT | 0.33 bp | 100.0% | 0.7 | 3.6M | 4.2% |

## Diagnostics

- Basis size selected: 257
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.044
- Maximum pairwise absolute correlation: 0.346
- Mean whole-market projection R²: 85.8%
- Median whole-market projection R²: 80.2%
- 10th-percentile whole-market projection R²: 74.2%
- Minimum whole-market projection R²: 71.3%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.7% | 1.4% | 0.0% | 0.0% |
| 5 | 6.4% | 2.9% | 0.7% | 0.1% |
| 10 | 8.5% | 4.6% | 2.0% | 0.5% |
| 15 | 10.7% | 6.3% | 3.2% | 1.2% |
| 20 | 12.9% | 7.9% | 4.7% | 2.2% |
| 25 | 15.3% | 10.0% | 6.0% | 3.4% |
| 30 | 17.5% | 11.6% | 7.5% | 4.5% |
| 35 | 19.6% | 13.3% | 9.0% | 6.3% |
| 40 | 21.6% | 14.9% | 10.3% | 7.3% |
| 45 | 23.6% | 16.5% | 11.8% | 8.2% |
| 50 | 25.5% | 18.0% | 13.2% | 9.3% |
| 55 | 27.6% | 19.9% | 14.9% | 11.6% |
| 60 | 29.6% | 21.5% | 16.5% | 13.2% |
| 65 | 31.6% | 23.3% | 17.9% | 14.0% |
| 70 | 33.5% | 24.8% | 19.3% | 16.3% |
| 75 | 35.3% | 26.3% | 20.9% | 17.3% |
| 80 | 37.1% | 27.7% | 22.4% | 19.2% |
| 85 | 39.1% | 29.7% | 23.9% | 20.7% |
| 90 | 41.0% | 31.5% | 25.5% | 22.0% |
| 95 | 42.6% | 32.8% | 27.0% | 23.6% |
| 100 | 44.4% | 34.6% | 28.4% | 25.4% |
| 105 | 46.0% | 35.9% | 29.8% | 26.7% |
| 110 | 47.7% | 37.8% | 31.6% | 28.2% |
| 115 | 49.5% | 39.5% | 32.8% | 29.0% |
| 120 | 51.1% | 40.9% | 34.5% | 30.6% |
| 125 | 52.8% | 42.5% | 36.1% | 31.9% |
| 130 | 54.5% | 44.2% | 37.5% | 33.4% |
| 135 | 56.0% | 46.0% | 39.2% | 34.7% |
| 140 | 57.5% | 47.2% | 40.8% | 36.6% |
| 145 | 59.0% | 48.9% | 42.1% | 38.3% |
| 150 | 60.4% | 50.3% | 43.7% | 39.9% |
| 155 | 61.9% | 51.7% | 45.2% | 42.0% |
| 160 | 63.3% | 53.4% | 46.7% | 43.5% |
| 165 | 64.7% | 54.8% | 47.9% | 44.9% |
| 170 | 66.1% | 56.6% | 49.6% | 46.3% |
| 175 | 67.4% | 58.0% | 50.9% | 47.7% |
| 180 | 68.6% | 59.3% | 52.5% | 48.6% |
| 185 | 69.9% | 60.5% | 54.0% | 49.9% |
| 190 | 71.2% | 62.1% | 55.4% | 51.9% |
| 195 | 72.4% | 63.7% | 56.8% | 53.7% |
| 200 | 73.6% | 65.1% | 58.2% | 54.4% |
| 205 | 74.8% | 66.4% | 59.9% | 55.6% |
| 210 | 76.0% | 67.8% | 61.3% | 57.8% |
| 215 | 77.2% | 69.2% | 62.8% | 59.5% |
| 220 | 78.3% | 70.6% | 64.0% | 60.6% |
| 225 | 79.4% | 71.8% | 65.6% | 62.2% |
| 230 | 80.5% | 73.0% | 66.9% | 64.1% |
| 235 | 81.5% | 74.2% | 68.3% | 65.6% |
| 240 | 82.6% | 75.7% | 69.5% | 67.1% |
| 245 | 83.6% | 77.0% | 71.0% | 68.4% |
| 250 | 84.5% | 78.3% | 72.2% | 69.6% |
| 255 | 85.4% | 79.5% | 73.5% | 70.4% |
| 257 | 85.8% | 80.2% | 74.2% | 71.3% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | XNO | BTTC | RIF | B2 | ONE | DEXE | AKE | BLESS | BROCCOLIF3B | DODO | SHAZ | AIA | CAT | B | QUICK | MIRA | MINIMAX | RECALL | STRAX | MMT | EWT | ID | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.058 | 0.059 | -0.034 | -0.051 | -0.061 | 0.010 | 0.032 | -0.024 | -0.035 | 0.009 | -0.019 | -0.002 | 0.050 | -0.028 | 0.020 | 0.039 | 0.024 | 0.115 | 0.060 | 0.076 | 0.077 | 0.059 | 0.056 |
| XNO | 0.058 | 1.000 | 0.050 | -0.075 | -0.005 | -0.053 | -0.040 | 0.011 | -0.059 | 0.009 | 0.010 | -0.009 | 0.043 | 0.049 | 0.027 | 0.083 | 0.003 | 0.005 | -0.019 | -0.007 | 0.009 | 0.021 | -0.026 | -0.011 |
| BTTC | 0.059 | 0.050 | 1.000 | -0.015 | 0.032 | 0.009 | 0.077 | -0.095 | -0.021 | 0.046 | -0.061 | -0.071 | 0.050 | -0.015 | 0.000 | 0.007 | 0.111 | 0.038 | 0.038 | 0.056 | 0.011 | -0.046 | 0.000 | 0.029 |
| RIF | -0.034 | -0.075 | -0.015 | 1.000 | 0.000 | 0.006 | -0.042 | 0.002 | 0.015 | 0.059 | -0.001 | -0.030 | 0.006 | -0.088 | -0.064 | -0.065 | -0.020 | -0.031 | -0.012 | -0.059 | -0.000 | 0.046 | 0.025 | -0.017 |
| B2 | -0.051 | -0.005 | 0.032 | 0.000 | 1.000 | -0.035 | -0.029 | -0.046 | -0.015 | 0.005 | 0.012 | -0.005 | 0.006 | -0.041 | -0.042 | 0.008 | -0.042 | 0.008 | -0.024 | 0.013 | 0.020 | -0.040 | 0.010 | 0.027 |
| ONE | -0.061 | -0.053 | 0.009 | 0.006 | -0.035 | 1.000 | 0.031 | -0.008 | -0.020 | 0.002 | 0.042 | -0.083 | -0.068 | 0.009 | -0.019 | 0.059 | -0.023 | -0.034 | -0.004 | 0.025 | 0.043 | 0.042 | 0.013 | 0.018 |
| DEXE | 0.010 | -0.040 | 0.077 | -0.042 | -0.029 | 0.031 | 1.000 | -0.012 | -0.019 | 0.046 | 0.021 | -0.020 | -0.025 | -0.008 | -0.058 | -0.015 | 0.046 | 0.032 | 0.013 | 0.013 | -0.008 | -0.057 | -0.017 | 0.029 |
| AKE | 0.032 | 0.011 | -0.095 | 0.002 | -0.046 | -0.008 | -0.012 | 1.000 | 0.070 | -0.068 | 0.051 | 0.016 | 0.003 | -0.022 | -0.039 | 0.028 | 0.015 | 0.023 | 0.040 | -0.056 | 0.074 | 0.005 | 0.031 | -0.026 |
| BLESS | -0.024 | -0.059 | -0.021 | 0.015 | -0.015 | -0.020 | -0.019 | 0.070 | 1.000 | 0.027 | 0.074 | 0.054 | -0.049 | 0.018 | -0.027 | 0.003 | -0.050 | -0.017 | -0.031 | 0.053 | 0.024 | 0.045 | -0.004 | -0.010 |
| BROCCOLIF3B | -0.035 | 0.009 | 0.046 | 0.059 | 0.005 | 0.002 | 0.046 | -0.068 | 0.027 | 1.000 | 0.010 | 0.016 | 0.025 | 0.020 | 0.030 | 0.006 | 0.023 | -0.078 | -0.000 | -0.033 | -0.076 | -0.040 | -0.005 | -0.032 |
| DODO | 0.009 | 0.010 | -0.061 | -0.001 | 0.012 | 0.042 | 0.021 | 0.051 | 0.074 | 0.010 | 1.000 | 0.004 | -0.013 | 0.014 | -0.029 | 0.045 | -0.035 | -0.062 | 0.037 | -0.015 | 0.070 | -0.012 | 0.054 | 0.078 |
| SHAZ | -0.019 | -0.009 | -0.071 | -0.030 | -0.005 | -0.083 | -0.020 | 0.016 | 0.054 | 0.016 | 0.004 | 1.000 | 0.046 | -0.007 | 0.002 | -0.024 | -0.015 | -0.030 | -0.024 | -0.031 | -0.025 | -0.006 | -0.020 | 0.003 |
| AIA | -0.002 | 0.043 | 0.050 | 0.006 | 0.006 | -0.068 | -0.025 | 0.003 | -0.049 | 0.025 | -0.013 | 0.046 | 1.000 | -0.014 | 0.057 | -0.001 | -0.025 | 0.017 | -0.026 | 0.032 | 0.021 | 0.006 | 0.047 | 0.049 |
| CAT | 0.050 | 0.049 | -0.015 | -0.088 | -0.041 | 0.009 | -0.008 | -0.022 | 0.018 | 0.020 | 0.014 | -0.007 | -0.014 | 1.000 | -0.031 | 0.049 | -0.000 | 0.008 | 0.066 | -0.015 | -0.003 | -0.003 | -0.001 | -0.028 |
| B | -0.028 | 0.027 | 0.000 | -0.064 | -0.042 | -0.019 | -0.058 | -0.039 | -0.027 | 0.030 | -0.029 | 0.002 | 0.057 | -0.031 | 1.000 | 0.012 | 0.011 | -0.017 | -0.016 | -0.031 | -0.027 | 0.014 | 0.012 | 0.060 |
| QUICK | 0.020 | 0.083 | 0.007 | -0.065 | 0.008 | 0.059 | -0.015 | 0.028 | 0.003 | 0.006 | 0.045 | -0.024 | -0.001 | 0.049 | 0.012 | 1.000 | 0.010 | -0.061 | 0.019 | -0.069 | -0.022 | 0.072 | 0.061 | -0.031 |
| MIRA | 0.039 | 0.003 | 0.111 | -0.020 | -0.042 | -0.023 | 0.046 | 0.015 | -0.050 | 0.023 | -0.035 | -0.015 | -0.025 | -0.000 | 0.011 | 0.010 | 1.000 | -0.018 | 0.058 | 0.005 | 0.038 | -0.033 | -0.024 | -0.030 |
| MINIMAX | 0.024 | 0.005 | 0.038 | -0.031 | 0.008 | -0.034 | 0.032 | 0.023 | -0.017 | -0.078 | -0.062 | -0.030 | 0.017 | 0.008 | -0.017 | -0.061 | -0.018 | 1.000 | 0.010 | 0.018 | -0.016 | -0.007 | -0.044 | 0.034 |
| RECALL | 0.115 | -0.019 | 0.038 | -0.012 | -0.024 | -0.004 | 0.013 | 0.040 | -0.031 | -0.000 | 0.037 | -0.024 | -0.026 | 0.066 | -0.016 | 0.019 | 0.058 | 0.010 | 1.000 | 0.020 | 0.021 | 0.016 | -0.047 | 0.008 |
| STRAX | 0.060 | -0.007 | 0.056 | -0.059 | 0.013 | 0.025 | 0.013 | -0.056 | 0.053 | -0.033 | -0.015 | -0.031 | 0.032 | -0.015 | -0.031 | -0.069 | 0.005 | 0.018 | 0.020 | 1.000 | 0.009 | 0.033 | -0.022 | 0.082 |
| MMT | 0.076 | 0.009 | 0.011 | -0.000 | 0.020 | 0.043 | -0.008 | 0.074 | 0.024 | -0.076 | 0.070 | -0.025 | 0.021 | -0.003 | -0.027 | -0.022 | 0.038 | -0.016 | 0.021 | 0.009 | 1.000 | -0.044 | 0.045 | 0.024 |
| EWT | 0.077 | 0.021 | -0.046 | 0.046 | -0.040 | 0.042 | -0.057 | 0.005 | 0.045 | -0.040 | -0.012 | -0.006 | 0.006 | -0.003 | 0.014 | 0.072 | -0.033 | -0.007 | 0.016 | 0.033 | -0.044 | 1.000 | 0.002 | 0.005 |
| ID | 0.059 | -0.026 | 0.000 | 0.025 | 0.010 | 0.013 | -0.017 | 0.031 | -0.004 | -0.005 | 0.054 | -0.020 | 0.047 | -0.001 | 0.012 | 0.061 | -0.024 | -0.044 | -0.047 | -0.022 | 0.045 | 0.002 | 1.000 | 0.040 |
| C | 0.056 | -0.011 | 0.029 | -0.017 | 0.027 | 0.018 | 0.029 | -0.026 | -0.010 | -0.032 | 0.078 | 0.003 | 0.049 | -0.028 | 0.060 | -0.031 | -0.030 | 0.034 | 0.008 | 0.082 | 0.024 | 0.005 | 0.040 | 1.000 |

The complete 257 × 257 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| WLFI | WLFIUSDT | 71.3% | 53.6% | QCOMBUSDT | -0.181 |
| SATS | 1000SATSUSDT | 71.3% | 53.5% | TURBOUSDT | 0.184 |
| BR | BRUSDT | 71.4% | 53.5% | FFUSDT | -0.190 |
| SFP | SFPUSDT | 71.4% | 53.5% | BTCUSDT | 0.239 |
| BZ | BZUSDT | 71.5% | 53.4% | PYRUSDT | 0.151 |
| CFG | CFGUSDT | 71.7% | 53.2% | BTCUSDT | 0.256 |
| DUSK | DUSKUSDT | 71.9% | 53.0% | KNCUSDT | 0.204 |
| ACM | ACMUSDT | 71.9% | 53.0% | XNYUSDT | 0.160 |
| RAY | RAYUSDT | 71.9% | 53.0% | BTCUSDT | 0.256 |
| LRCX | LRCXUSDT | 72.0% | 53.0% | CIENUSDT | 0.230 |
| STABLE | STABLEUSDT | 72.0% | 52.9% | AUDIOUSDT | -0.167 |
| BOT | BOTUSDT | 72.0% | 52.9% | ZKUSDT | 0.174 |
| H | HUSDT | 72.1% | 52.9% | YFIUSDT | 0.178 |
| PLUME | PLUMEUSDT | 72.1% | 52.8% | BTCUSDT | 0.243 |
| DYDX | DYDXUSDT | 72.2% | 52.8% | BTCUSDT | 0.217 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

