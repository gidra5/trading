# Binance portfolio basis

Generated 2026-07-23T19:09:52.340Z.

## Scope

- Products: all
- Product listings discovered: spot 0, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 2301, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-22T18:00Z through 2026-07-22T23:59Z
- Sampling: exactly 360 1m log returns (0.3 days)
- Correlation: pearson
- Pivot tie-break: among candidates within 1.0% of the maximum unexplained variance, select the largest mean absolute 1m return
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 2426 listings, 2301 active listings, 795 deduplicated economic assets
- Return universe: 668 eligible assets from 669 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Orthogonality remains primary; mean absolute candle return only chooses among near-equivalent residual candidates at this scale. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max abs corr to earlier | Closest earlier | Mean abs return | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (usdm-futures) | usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 2.51 bp | 100.0% | 0.1 | 1.1B | 25.1% |
| 2 | B2 | B2USDT (usdm-futures) | usdm-futures | 99.8% | 0.060 | BTCUSDT | 49.97 bp | 100.0% | 0 | 13M | 615.2% |
| 3 | RIF | RIFUSDT (usdm-futures) | usdm-futures | 99.8% | 0.055 | BTCUSDT | 47.53 bp | 100.0% | 0 | 25.7M | 485.0% |
| 4 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.8% | 0.046 | B2USDT | 31.24 bp | 100.0% | 0 | 26.5M | 301.7% |
| 5 | ON | ONUSDT (usdm-futures) | usdm-futures | 99.6% | 0.055 | B2USDT | 31.09 bp | 100.0% | 0 | 13.6M | 295.6% |
| 6 | DEXE | DEXEUSDT (usdm-futures) | usdm-futures | 100.0% | 0.021 | ONUSDT | 29.78 bp | 100.0% | 0 | 40.7M | 298.4% |
| 7 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 99.7% | 0.070 | AKEUSDT | 26.24 bp | 100.0% | 0 | 5.3M | 264.9% |
| 8 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 99.7% | 0.052 | RIFUSDT | 18.21 bp | 100.0% | 0 | 4.8M | 249.9% |
| 9 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 99.5% | 0.068 | AIAUSDT | 17.64 bp | 100.0% | 0 | 571.6K | 169.9% |
| 10 | RE | REUSDT (usdm-futures) | usdm-futures | 99.5% | 0.075 | BLESSUSDT | 17.64 bp | 100.0% | 0 | 40.3M | 170.6% |
| 11 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 99.5% | 0.061 | AIAUSDT | 13.70 bp | 100.0% | 0.1 | 1.5M | 154.8% |
| 12 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 99.7% | 0.045 | BLESSUSDT | 11.45 bp | 100.0% | 0 | 240.2K | 119.8% |
| 13 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 99.6% | 0.064 | AGTUSDT | 10.62 bp | 100.0% | 0 | 9.1M | 185.1% |
| 14 | MINIMAX | MINIMAXUSDT (usdm-futures) | usdm-futures | 99.7% | 0.035 | XNYUSDT | 9.03 bp | 100.0% | 0.1 | 845.7K | 129.0% |
| 15 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 99.3% | 0.056 | RIFUSDT | 7.28 bp | 100.0% | 0.1 | 111.2K | 76.0% |
| 16 | BEL | BELUSDT (usdm-futures) | usdm-futures | 99.2% | 0.059 | B2USDT | 6.93 bp | 100.0% | 0 | 522.9K | 74.0% |
| 17 | HPE | HPEUSDT (usdm-futures) | usdm-futures | 99.2% | 0.063 | AGTUSDT | 6.02 bp | 100.0% | 0 | 225K | 80.7% |
| 18 | BNC | BNCUSDT (usdm-futures) | usdm-futures | 99.0% | 0.073 | MINIMAXUSDT | 5.50 bp | 100.0% | 0.2 | 80.9K | 73.6% |
| 19 | LLY | LLYUSDT (usdm-futures) | usdm-futures | 99.3% | 0.063 | BELUSDT | 1.58 bp | 100.0% | 0.1 | 118.3K | 29.4% |
| 20 | DIS | DISUSDT (usdm-futures) | usdm-futures | 98.5% | 0.085 | ONUSDT | 2.03 bp | 100.0% | 0.1 | 53.9K | 34.7% |
| 21 | US | USUSDT (usdm-futures) | usdm-futures | 98.4% | 0.107 | BTCUSDT | 13.37 bp | 100.0% | 0 | 3.4M | 137.5% |
| 22 | COST | COSTUSDT (usdm-futures) | usdm-futures | 98.8% | 0.076 | DEXEUSDT | 1.05 bp | 100.0% | 0.1 | 90.2K | 21.6% |
| 23 | DODOX | DODOXUSDT (usdm-futures) | usdm-futures | 98.0% | 0.076 | DEXEUSDT | 21.05 bp | 100.0% | 0 | 6.9M | 206.8% |
| 24 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 97.8% | 0.076 | ONUSDT | 14.53 bp | 100.0% | 0 | 962.6K | 140.2% |
| 25 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 97.9% | 0.110 | XNYUSDT | 8.94 bp | 100.0% | 0.1 | 1.2M | 126.1% |
| 26 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 97.9% | 0.100 | BTCUSDT | 7.55 bp | 100.0% | 0.1 | 172.5K | 77.6% |
| 27 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 97.3% | 0.120 | SPORTFUNUSDT | 8.84 bp | 100.0% | 0.1 | 81.9K | 89.5% |
| 28 | NXPC | NXPCUSDT (usdm-futures) | usdm-futures | 97.3% | 0.085 | ONUSDT | 8.83 bp | 100.0% | 0.1 | 574.4K | 99.7% |
| 29 | BAN | BANUSDT (usdm-futures) | usdm-futures | 97.6% | 0.120 | DEXEUSDT | 6.48 bp | 100.0% | 0.1 | 2M | 114.6% |
| 30 | IBM | IBMUSDT (usdm-futures) | usdm-futures | 96.9% | 0.101 | BTCUSDT | 11.13 bp | 100.0% | 0 | 32.5M | 171.6% |
| 31 | ROBO | ROBOUSDT (usdm-futures) | usdm-futures | 96.9% | 0.104 | BELUSDT | 6.77 bp | 100.0% | 0.1 | 438.4K | 89.6% |
| 32 | ESP | ESPUSDT (usdm-futures) | usdm-futures | 96.8% | 0.108 | ONUSDT | 5.57 bp | 100.0% | 0.1 | 449K | 56.0% |
| 33 | FLEX | FLEXUSDT (usdm-futures) | usdm-futures | 96.9% | 0.083 | BNCUSDT | 2.61 bp | 100.0% | 0 | 74.1K | 65.8% |
| 34 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 96.8% | 0.102 | DEXEUSDT | 2.43 bp | 100.0% | 0.1 | 2.9M | 31.8% |
| 35 | XLE | XLEUSDT (usdm-futures) | usdm-futures | 97.2% | 0.094 | GWEIUSDT | 2.01 bp | 100.0% | 0.1 | 19.9K | 31.3% |
| 36 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 96.2% | 0.137 | AGTUSDT | 17.64 bp | 100.0% | 0 | 1.5M | 203.8% |
| 37 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 96.3% | 0.131 | DISUSDT | 13.40 bp | 100.0% | 0 | 3.3M | 129.0% |
| 38 | HMSTR | HMSTRUSDT (usdm-futures) | usdm-futures | 95.9% | 0.116 | BULLAUSDT | 8.38 bp | 100.0% | 0.1 | 960.3K | 92.6% |
| 39 | BOB | 1000000BOBUSDT (usdm-futures) | usdm-futures | 95.8% | 0.117 | MINIMAXUSDT | 6.22 bp | 100.0% | 0.2 | 72.3K | 67.2% |
| 40 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 95.8% | 0.091 | MINIMAXUSDT | 5.73 bp | 100.0% | 0.1 | 42.2K | 58.4% |
| 41 | ASR | ASRUSDT (usdm-futures) | usdm-futures | 95.5% | 0.127 | DEXEUSDT | 4.94 bp | 100.0% | 0.3 | 53.2K | 58.7% |
| 42 | PANW | PANWUSDT (usdm-futures) | usdm-futures | 95.5% | 0.108 | AKEUSDT | 4.62 bp | 100.0% | 0 | 74.9K | 75.7% |
| 43 | ASML | ASMLUSDT (usdm-futures) | usdm-futures | 95.7% | 0.136 | HPEUSDT | 4.03 bp | 100.0% | 0 | 603.4K | 46.2% |
| 44 | CIEN | CIENUSDT (usdm-futures) | usdm-futures | 95.1% | 0.142 | AKEUSDT | 4.05 bp | 100.0% | 0 | 54.8K | 76.1% |
| 45 | TAC | TACUSDT (usdm-futures) | usdm-futures | 94.9% | 0.185 | USUSDT | 14.20 bp | 100.0% | 0 | 1.2M | 151.6% |
| 46 | HK1810 | HK1810USDT (usdm-futures) | usdm-futures | 95.2% | 0.106 | BULLAUSDT | 4.01 bp | 100.0% | 0.1 | 119.4K | 54.8% |
| 47 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 94.3% | 0.122 | EVAAUSDT | 13.28 bp | 100.0% | 0.1 | 4M | 129.5% |
| 48 | MITO | MITOUSDT (usdm-futures) | usdm-futures | 94.6% | 0.131 | ONUSDT | 11.56 bp | 100.0% | 0.1 | 586K | 121.8% |
| 49 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 94.5% | 0.120 | ICNTUSDT | 11.29 bp | 100.0% | 0.1 | 608.9K | 113.5% |
| 50 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 94.2% | 0.141 | DEXEUSDT | 5.74 bp | 100.0% | 0.1 | 503K | 86.5% |
| 51 | BX | BXUSDT (usdm-futures) | usdm-futures | 94.5% | 0.105 | AIAUSDT | 3.13 bp | 100.0% | 0 | 581K | 40.4% |
| 52 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 94.5% | 0.134 | LLYUSDT | 2.49 bp | 100.0% | 0 | 186K | 42.4% |
| 53 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 93.6% | 0.173 | XLEUSDT | 13.60 bp | 100.0% | 0 | 1.2M | 171.5% |
| 54 | STO | STOUSDT (usdm-futures) | usdm-futures | 93.6% | 0.128 | BTCUSDT | 7.68 bp | 100.0% | 0.1 | 520K | 82.4% |
| 55 | GENIUS | GENIUSUSDT (usdm-futures) | usdm-futures | 93.8% | 0.115 | BTCUSDT | 6.83 bp | 100.0% | 0.1 | 450.8K | 73.5% |
| 56 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 93.9% | 0.163 | BTCUSDT | 3.76 bp | 100.0% | 0.1 | 123K | 42.8% |
| 57 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 93.2% | 0.165 | AGTUSDT | 16.76 bp | 100.0% | 0 | 1.7M | 159.7% |
| 58 | RESOLV | RESOLVUSDT (usdm-futures) | usdm-futures | 92.9% | 0.151 | BTCUSDT | 7.18 bp | 100.0% | 0.1 | 282.1K | 71.7% |
| 59 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 92.9% | 0.142 | BTCUSDT | 6.07 bp | 100.0% | 0.1 | 172.6K | 67.1% |
| 60 | MOCA | MOCAUSDT (usdm-futures) | usdm-futures | 92.6% | 0.115 | HMSTRUSDT | 4.93 bp | 100.0% | 0.3 | 35.6K | 59.1% |
| 61 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 93.0% | 0.121 | NXPCUSDT | 4.73 bp | 100.0% | 0 | 5.4M | 53.9% |
| 62 | SAPIEN | SAPIENUSDT (usdm-futures) | usdm-futures | 92.2% | 0.168 | BTCUSDT | 20.12 bp | 100.0% | 0 | 2.7M | 221.3% |
| 63 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 92.0% | 0.139 | EVAAUSDT | 13.52 bp | 100.0% | 0 | 499.2K | 153.5% |
| 64 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 91.7% | 0.139 | ONUSDT | 11.22 bp | 100.0% | 0.1 | 369.5K | 120.3% |
| 65 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 91.5% | 0.138 | ICNTUSDT | 10.69 bp | 100.0% | 0 | 294.3K | 112.4% |
| 66 | USELESS | USELESSUSDT (usdm-futures) | usdm-futures | 91.6% | 0.196 | BTCUSDT | 10.00 bp | 100.0% | 0 | 1.3M | 144.3% |
| 67 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 91.5% | 0.130 | BTCUSDT | 6.11 bp | 100.0% | 0.1 | 63.1K | 62.1% |
| 68 | XAUT | XAUTUSDT (usdm-futures) | usdm-futures | 91.6% | 0.135 | RIFUSDT | 1.45 bp | 100.0% | 0.1 | 6.3M | 17.9% |
| 69 | BANK | BANKUSDT (usdm-futures) | usdm-futures | 91.0% | 0.141 | BTCUSDT | 47.66 bp | 100.0% | 0 | 266.7M | 501.7% |
| 70 | AT | ATUSDT (usdm-futures) | usdm-futures | 91.0% | 0.118 | MINIMAXUSDT | 4.88 bp | 100.0% | 0.1 | 119.5K | 53.1% |
| 71 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 91.3% | 0.146 | DISUSDT | 1.28 bp | 100.0% | 0.1 | 44.4K | 25.9% |
| 72 | B | BUSDT (usdm-futures) | usdm-futures | 90.6% | 0.146 | CAPUSDT | 10.92 bp | 100.0% | 0.1 | 1.8M | 127.3% |
| 73 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 90.3% | 0.147 | AKEUSDT | 7.83 bp | 100.0% | 0.1 | 795.5K | 89.4% |
| 74 | WEN | WENUSDT (usdm-futures) | usdm-futures | 90.6% | 0.109 | ICNTUSDT | 3.78 bp | 100.0% | 0.1 | 120K | 65.3% |
| 75 | MMT | MMTUSDT (usdm-futures) | usdm-futures | 89.6% | 0.115 | AGTUSDT | 7.00 bp | 100.0% | 0.1 | 487.9K | 79.0% |
| 76 | ZM | ZMUSDT (usdm-futures) | usdm-futures | 90.0% | 0.119 | BLUAIUSDT | 3.46 bp | 100.0% | 0 | 70.1K | 53.0% |
| 77 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 89.2% | 0.167 | GWEIUSDT | 15.77 bp | 100.0% | 0.1 | 695.3K | 182.0% |
| 78 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 89.1% | 0.146 | STOUSDT | 15.70 bp | 100.0% | 0 | 3.3M | 157.8% |
| 79 | OPN | OPNUSDT (usdm-futures) | usdm-futures | 89.2% | 0.162 | BTCUSDT | 14.60 bp | 100.0% | 0 | 2.9M | 148.4% |
| 80 | XAN | XANUSDT (usdm-futures) | usdm-futures | 89.0% | 0.121 | RIFUSDT | 9.09 bp | 100.0% | 0 | 303.2K | 112.4% |
| 81 | NMR | NMRUSDT (usdm-futures) | usdm-futures | 88.9% | 0.208 | BTCUSDT | 3.00 bp | 100.0% | 0.1 | 160.9K | 32.3% |
| 82 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 88.8% | 0.128 | BTCUSDT | 2.81 bp | 100.0% | 0.2 | 127.8K | 34.0% |
| 83 | STXX | STXXUSDT (usdm-futures) | usdm-futures | 88.7% | 0.148 | ASMLUSDT | 8.03 bp | 100.0% | 0.1 | 427K | 100.3% |
| 84 | RIVN | RIVNUSDT (usdm-futures) | usdm-futures | 88.6% | 0.139 | B2USDT | 3.48 bp | 100.0% | 0.1 | 164.6K | 59.1% |
| 85 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 88.5% | 0.136 | MITOUSDT | 3.09 bp | 100.0% | 0 | 88.5K | 44.4% |
| 86 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 88.3% | 0.160 | IBMUSDT | 6.74 bp | 100.0% | 0 | 448.9K | 70.7% |
| 87 | GOOGL | GOOGLUSDT (usdm-futures) | usdm-futures | 88.0% | 0.156 | BTCUSDT | 13.40 bp | 100.0% | 0 | 257.8M | 209.8% |
| 88 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 88.4% | 0.131 | B2USDT | 1.85 bp | 100.0% | 0.1 | 54.7K | 32.7% |
| 89 | HOME | HOMEUSDT (usdm-futures) | usdm-futures | 87.7% | 0.168 | STOUSDT | 11.53 bp | 100.0% | 0 | 3.5M | 135.2% |
| 90 | ZBT | ZBTUSDT (usdm-futures) | usdm-futures | 87.4% | 0.179 | ESPUSDT | 8.57 bp | 100.0% | 0 | 1.1M | 86.8% |
| 91 | SLP | SLPUSDT (usdm-futures) | usdm-futures | 87.6% | 0.163 | DATAIPUSDT | 6.51 bp | 100.0% | 0.1 | 108.6K | 65.4% |
| 92 | AWE | AWEUSDT (usdm-futures) | usdm-futures | 87.6% | 0.124 | BNCUSDT | 5.90 bp | 100.0% | 0.1 | 201.4K | 66.6% |
| 93 | TOWNS | TOWNSUSDT (usdm-futures) | usdm-futures | 86.9% | 0.181 | CSCOUSDT | 13.35 bp | 100.0% | 0.1 | 796.1K | 150.3% |
| 94 | NOK | NOKUSDT (usdm-futures) | usdm-futures | 86.7% | 0.132 | HPEUSDT | 10.62 bp | 100.0% | 0.1 | 2.8M | 133.8% |
| 95 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 86.6% | 0.139 | CYSUSDT | 10.38 bp | 100.0% | 0 | 246.9K | 101.3% |
| 96 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 86.9% | 0.139 | BXUSDT | 4.09 bp | 100.0% | 0.1 | 209K | 49.9% |
| 97 | BAS | BASUSDT (usdm-futures) | usdm-futures | 86.0% | 0.148 | 1000000BOBUSDT | 17.47 bp | 100.0% | 0 | 1.8M | 164.4% |
| 98 | SPK | SPKUSDT (usdm-futures) | usdm-futures | 86.1% | 0.160 | DATAIPUSDT | 3.52 bp | 100.0% | 0.2 | 325.2K | 52.7% |
| 99 | EWJ | EWJUSDT (usdm-futures) | usdm-futures | 86.2% | 0.138 | LLYUSDT | 1.30 bp | 100.0% | 0.1 | 179.1K | 17.6% |
| 100 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 85.4% | 0.129 | SPORTFUNUSDT | 9.94 bp | 100.0% | 0 | 897.7K | 97.0% |
| 101 | HEMI | HEMIUSDT (usdm-futures) | usdm-futures | 85.3% | 0.252 | BTCUSDT | 11.30 bp | 100.0% | 0 | 1.5M | 110.9% |
| 102 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 85.3% | 0.183 | BTCUSDT | 8.45 bp | 100.0% | 0.1 | 454.8K | 93.5% |
| 103 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 85.2% | 0.136 | BTCUSDT | 8.29 bp | 100.0% | 0.1 | 157.5K | 79.3% |
| 104 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 84.7% | 0.129 | FLEXUSDT | 10.37 bp | 100.0% | 0.1 | 222.7K | 114.6% |
| 105 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 84.8% | 0.175 | XPINUSDT | 6.30 bp | 100.0% | 0.1 | 174.1K | 103.9% |
| 106 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 84.9% | 0.125 | USUSDT | 6.14 bp | 100.0% | 0.1 | 617.6K | 80.9% |
| 107 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 84.7% | 0.189 | BTCUSDT | 5.80 bp | 100.0% | 0.1 | 208.8K | 73.5% |
| 108 | PROM | PROMUSDT (usdm-futures) | usdm-futures | 84.1% | 0.162 | SAPIENUSDT | 14.59 bp | 100.0% | 0.1 | 1.4M | 153.3% |
| 109 | O | OUSDT (usdm-futures) | usdm-futures | 83.8% | 0.170 | NOKUSDT | 8.71 bp | 100.0% | 0 | 871.8K | 99.4% |
| 110 | STAR | STARUSDT (usdm-futures) | usdm-futures | 83.7% | 0.201 | MMTUSDT | 16.08 bp | 100.0% | 0 | 414.6K | 176.9% |
| 111 | BOT | BOTUSDT (usdm-futures) | usdm-futures | 83.6% | 0.172 | ASRUSDT | 10.57 bp | 100.0% | 0.1 | 433K | 117.7% |
| 112 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 83.7% | 0.209 | GENIUSUSDT | 6.31 bp | 100.0% | 0.1 | 1.1M | 95.2% |
| 113 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 83.5% | 0.220 | BTCUSDT | 7.61 bp | 100.0% | 0.1 | 359.5K | 80.7% |
| 114 | SYN | SYNUSDT (usdm-futures) | usdm-futures | 83.0% | 0.151 | SAFEUSDT | 11.30 bp | 100.0% | 0.1 | 2.7M | 111.6% |
| 115 | DIA | DIAUSDT (usdm-futures) | usdm-futures | 83.2% | 0.174 | HPEUSDT | 5.81 bp | 100.0% | 0.1 | 76.9K | 62.9% |
| 116 | ENSO | ENSOUSDT (usdm-futures) | usdm-futures | 83.0% | 0.175 | BTCUSDT | 4.91 bp | 100.0% | 0.1 | 512.1K | 52.4% |
| 117 | GPS | GPSUSDT (usdm-futures) | usdm-futures | 83.0% | 0.158 | TAKEUSDT | 4.75 bp | 100.0% | 0.1 | 187.4K | 57.1% |
| 118 | TER | TERUSDT (usdm-futures) | usdm-futures | 82.8% | 0.142 | MINIMAXUSDT | 3.96 bp | 100.0% | 0.1 | 85.3K | 75.0% |
| 119 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 82.8% | 0.182 | GWEIUSDT | 2.31 bp | 100.0% | 0 | 61.3K | 41.4% |
| 120 | TENCENT | TENCENTUSDT (usdm-futures) | usdm-futures | 82.2% | 0.144 | STBLUSDT | 3.21 bp | 100.0% | 0.1 | 90.3K | 40.3% |
| 121 | AIGENSYN | AIGENSYNUSDT (usdm-futures) | usdm-futures | 81.9% | 0.205 | BTCUSDT | 7.04 bp | 100.0% | 0.1 | 864.9K | 77.8% |
| 122 | JOE | JOEUSDT (usdm-futures) | usdm-futures | 81.9% | 0.203 | BTCUSDT | 6.85 bp | 100.0% | 0.1 | 239.6K | 80.2% |
| 123 | CRM | CRMUSDT (usdm-futures) | usdm-futures | 81.9% | 0.160 | MINIMAXUSDT | 5.75 bp | 100.0% | 0 | 202.6K | 93.3% |
| 124 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 81.4% | 0.118 | WENUSDT | 8.89 bp | 100.0% | 0.1 | 586.3K | 101.5% |
| 125 | ALPINE | ALPINEUSDT (usdm-futures) | usdm-futures | 81.2% | 0.195 | BTCUSDT | 6.50 bp | 100.0% | 0.1 | 121.1K | 68.7% |
| 126 | INX | INXUSDT (usdm-futures) | usdm-futures | 80.9% | 0.169 | XAUTUSDT | 7.87 bp | 100.0% | 0.1 | 261K | 83.5% |
| 127 | KMNO | KMNOUSDT (usdm-futures) | usdm-futures | 80.9% | 0.194 | BTCUSDT | 2.96 bp | 100.0% | 0.1 | 103.4K | 36.4% |
| 128 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 81.1% | 0.183 | CTRUSDT | 0.89 bp | 100.0% | 0.1 | 55.9K | 24.0% |
| 129 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 80.4% | 0.157 | OUSDT | 13.18 bp | 100.0% | 0.1 | 804.2K | 136.1% |
| 130 | SPCX | SPCXUSDT (usdm-futures) | usdm-futures | 80.2% | 0.226 | USELESSUSDT | 6.87 bp | 100.0% | 0 | 297.3M | 71.4% |
| 131 | MELANIA | MELANIAUSDT (usdm-futures) | usdm-futures | 79.9% | 0.150 | BTCUSDT | 4.67 bp | 100.0% | 0.1 | 109.2K | 54.1% |
| 132 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 80.3% | 0.153 | ZESTUSDT | 1.22 bp | 100.0% | 0 | 115.8K | 17.0% |
| 133 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 79.4% | 0.174 | AMZNUSDT | 9.42 bp | 100.0% | 0.1 | 278.3K | 114.3% |
| 134 | HD | HDUSDT (usdm-futures) | usdm-futures | 79.5% | 0.164 | CRMUSDT | 1.89 bp | 100.0% | 0 | 69.8K | 31.6% |
| 135 | NIGHT | NIGHTUSDT (usdm-futures) | usdm-futures | 78.4% | 0.181 | KSTRUSDT | 18.06 bp | 100.0% | 0 | 6M | 181.0% |
| 136 | CRWV | CRWVUSDT (usdm-futures) | usdm-futures | 78.8% | 0.180 | GOOGLUSDT | 13.00 bp | 100.0% | 0 | 2.4M | 162.7% |
| 137 | ARPA | ARPAUSDT (usdm-futures) | usdm-futures | 78.4% | 0.256 | BTCUSDT | 6.50 bp | 100.0% | 0.1 | 454.7K | 71.7% |
| 138 | WLFI | WLFIUSDT (usdm-futures) | usdm-futures | 78.4% | 0.175 | SAPIENUSDT | 5.45 bp | 100.0% | 0 | 6M | 54.6% |
| 139 | THE | THEUSDT (usdm-futures) | usdm-futures | 78.2% | 0.184 | USELESSUSDT | 8.02 bp | 100.0% | 0 | 851.4K | 79.7% |
| 140 | PORTAL | PORTALUSDT (usdm-futures) | usdm-futures | 78.4% | 0.217 | BTCUSDT | 5.36 bp | 100.0% | 0.2 | 213.5K | 61.0% |
| 141 | EPIC | EPICUSDT (usdm-futures) | usdm-futures | 77.9% | 0.151 | REUSDT | 15.56 bp | 100.0% | 0 | 3.3M | 154.5% |
| 142 | ACE | ACEUSDT (usdm-futures) | usdm-futures | 77.5% | 0.139 | TOWNSUSDT | 15.43 bp | 100.0% | 0 | 3.7M | 157.7% |
| 143 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 77.5% | 0.160 | SPKUSDT | 8.77 bp | 100.0% | 0.1 | 156.1K | 87.9% |
| 144 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 77.7% | 0.196 | BTCUSDT | 7.27 bp | 100.0% | 0 | 277.3K | 69.7% |
| 145 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 77.1% | 0.175 | PROMUSDT | 4.77 bp | 100.0% | 0.1 | 131K | 48.7% |
| 146 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 76.4% | 0.194 | BTCUSDT | 10.74 bp | 100.0% | 0 | 1.3M | 105.6% |
| 147 | ALICE | ALICEUSDT (usdm-futures) | usdm-futures | 76.4% | 0.159 | SLPUSDT | 6.58 bp | 100.0% | 0.1 | 424.9K | 68.0% |
| 148 | PUMP | PUMPUSDT (usdm-futures) | usdm-futures | 76.2% | 0.337 | BTCUSDT | 9.45 bp | 100.0% | 0.1 | 17M | 102.9% |
| 149 | M | MUSDT (usdm-futures) | usdm-futures | 76.4% | 0.195 | TAKEUSDT | 6.22 bp | 100.0% | 0.1 | 517K | 64.1% |
| 150 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 75.9% | 0.168 | TACUSDT | 13.33 bp | 100.0% | 0 | 1.9M | 131.7% |
| 151 | OPENAI | OPENAIUSDT (usdm-futures) | usdm-futures | 75.6% | 0.206 | ANTHROPICUSDT | 8.25 bp | 100.0% | 0 | 1.2M | 110.9% |
| 152 | ERA | ERAUSDT (usdm-futures) | usdm-futures | 75.4% | 0.157 | HDUSDT | 24.46 bp | 100.0% | 0 | 18.4M | 241.4% |
| 153 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 75.3% | 0.127 | GWEIUSDT | 14.52 bp | 100.0% | 0 | 2.1M | 165.7% |
| 154 | OPEN | OPENUSDT (usdm-futures) | usdm-futures | 75.5% | 0.222 | BTCUSDT | 6.20 bp | 100.0% | 0.1 | 379.3K | 68.8% |
| 155 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 75.3% | 0.191 | PANWUSDT | 6.11 bp | 100.0% | 0.1 | 63.6K | 96.9% |
| 156 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 75.2% | 0.151 | HEMIUSDT | 5.26 bp | 100.0% | 0 | 167.4K | 60.9% |
| 157 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 74.7% | 0.180 | 龙虾USDT | 9.47 bp | 100.0% | 0.1 | 459K | 110.5% |
| 158 | TST | TSTUSDT (usdm-futures) | usdm-futures | 74.7% | 0.167 | TACUSDT | 6.97 bp | 100.0% | 0.1 | 213K | 97.2% |
| 159 | BMNR | BMNRUSDT (usdm-futures) | usdm-futures | 74.5% | 0.275 | BTCUSDT | 6.16 bp | 100.0% | 0.1 | 423K | 74.2% |
| 160 | BSV | BSVUSDT (usdm-futures) | usdm-futures | 74.4% | 0.197 | ACUUSDT | 3.76 bp | 100.0% | 0.2 | 167.2K | 46.1% |
| 161 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 73.7% | 0.201 | USUSDT | 25.51 bp | 100.0% | 0 | 12.8M | 255.8% |
| 162 | CGPT | CGPTUSDT (usdm-futures) | usdm-futures | 73.3% | 0.162 | SPACEUSDT | 10.07 bp | 100.0% | 0.1 | 502.2K | 112.8% |
| 163 | BANANA | BANANAUSDT (usdm-futures) | usdm-futures | 73.2% | 0.178 | MOCAUSDT | 9.36 bp | 100.0% | 0.1 | 469.5K | 103.1% |
| 164 | WET | WETUSDT (usdm-futures) | usdm-futures | 73.3% | 0.171 | CLOUSDT | 5.18 bp | 100.0% | 0.1 | 199K | 62.6% |
| 165 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 73.0% | 0.223 | BTCUSDT | 8.15 bp | 100.0% | 0 | 529.9K | 92.9% |
| 166 | BZ | BZUSDT (usdm-futures) | usdm-futures | 73.2% | 0.136 | XLEUSDT | 3.44 bp | 100.0% | 0.1 | 82.8M | 36.9% |
| 167 | SPELL | SPELLUSDT (usdm-futures) | usdm-futures | 72.6% | 0.154 | SPACEUSDT | 12.61 bp | 100.0% | 0.1 | 2.3M | 136.8% |
| 168 | FWDI | FWDIUSDT (usdm-futures) | usdm-futures | 72.1% | 0.157 | BTCUSDT | 6.84 bp | 100.0% | 0.1 | 141.3K | 87.5% |
| 169 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 72.2% | 0.161 | BLUAIUSDT | 6.25 bp | 100.0% | 0.1 | 233.5K | 61.6% |
| 170 | AIN | AINUSDT (usdm-futures) | usdm-futures | 71.5% | 0.150 | TOWNSUSDT | 10.51 bp | 100.0% | 0 | 112.9K | 102.9% |
| 171 | TA | TAUSDT (usdm-futures) | usdm-futures | 71.5% | 0.174 | LYNUSDT | 7.84 bp | 100.0% | 0.1 | 455.1K | 81.6% |
| 172 | BMT | BMTUSDT (usdm-futures) | usdm-futures | 71.5% | 0.176 | FIGHTUSDT | 6.23 bp | 100.0% | 0.1 | 110.5K | 66.6% |
| 173 | IRYS | IRYSUSDT (usdm-futures) | usdm-futures | 71.4% | 0.285 | BTCUSDT | 5.38 bp | 100.0% | 0.1 | 195.7K | 60.9% |
| 174 | BABY | BABYUSDT (usdm-futures) | usdm-futures | 71.7% | 0.212 | BTCUSDT | 4.96 bp | 100.0% | 0.2 | 246.2K | 56.5% |
| 175 | TREE | TREEUSDT (usdm-futures) | usdm-futures | 70.9% | 0.347 | BTCUSDT | 6.84 bp | 100.0% | 0 | 560.2K | 69.3% |
| 176 | INIT | INITUSDT (usdm-futures) | usdm-futures | 70.8% | 0.206 | BSVUSDT | 5.12 bp | 100.0% | 0.1 | 262.5K | 56.3% |
| 177 | ZHIPU | ZHIPUUSDT (usdm-futures) | usdm-futures | 70.1% | 0.206 | ALPINEUSDT | 11.67 bp | 100.0% | 0 | 4.4M | 128.0% |
| 178 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 70.1% | 0.144 | BTCUSDT | 11.13 bp | 100.0% | 0 | 289.3K | 106.6% |
| 179 | LSK | LSKUSDT (usdm-futures) | usdm-futures | 70.0% | 0.208 | DIAUSDT | 4.77 bp | 100.0% | 0.3 | 104.7K | 61.0% |
| 180 | GAS | GASUSDT (usdm-futures) | usdm-futures | 70.2% | 0.221 | BTCUSDT | 4.41 bp | 100.0% | 0.2 | 197.8K | 50.6% |
| 181 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 69.7% | 0.249 | BTCUSDT | 7.82 bp | 100.0% | 0 | 4.6M | 76.3% |
| 182 | NVDA | NVDAUSDT (usdm-futures) | usdm-futures | 69.4% | 0.235 | ASMLUSDT | 4.49 bp | 100.0% | 0 | 28.2M | 46.7% |
| 183 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 69.6% | 0.177 | ARIAUSDT | 3.16 bp | 100.0% | 0.1 | 416.7K | 42.6% |
| 184 | FLUID | FLUIDUSDT (usdm-futures) | usdm-futures | 68.6% | 0.166 | TACUSDT | 8.40 bp | 100.0% | 0.1 | 103.1K | 90.7% |
| 185 | EDEN | EDENUSDT (usdm-futures) | usdm-futures | 68.5% | 0.321 | BTCUSDT | 6.96 bp | 100.0% | 0 | 431.9K | 69.9% |
| 186 | GIGGLE | GIGGLEUSDT (usdm-futures) | usdm-futures | 68.7% | 0.301 | BTCUSDT | 5.11 bp | 100.0% | 0.1 | 1.1M | 58.0% |
| 187 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 68.4% | 0.190 | GOOGLUSDT | 2.50 bp | 100.0% | 0 | 114.9K | 40.4% |
| 188 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 67.9% | 0.210 | BTCUSDT | 7.64 bp | 100.0% | 0.1 | 346.4K | 83.7% |
| 189 | 4 | 4USDT (usdm-futures) | usdm-futures | 67.8% | 0.193 | BMTUSDT | 7.25 bp | 100.0% | 0 | 156.4K | 82.3% |
| 190 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 67.9% | 0.185 | ZEREBROUSDT | 6.62 bp | 100.0% | 0.1 | 413.1K | 68.8% |
| 191 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 67.7% | 0.233 | BTCUSDT | 6.50 bp | 100.0% | 0.1 | 264.4K | 75.3% |
| 192 | HOLO | HOLOUSDT (usdm-futures) | usdm-futures | 67.7% | 0.180 | CIENUSDT | 5.16 bp | 100.0% | 0.1 | 326.1K | 52.9% |
| 193 | CAT | CATUSDT (usdm-futures) | usdm-futures | 67.4% | 0.196 | MOCAUSDT | 2.48 bp | 100.0% | 0 | 45.2K | 45.6% |
| 194 | 我踏马来了 | 我踏马来了USDT (usdm-futures) | usdm-futures | 67.3% | 0.177 | BTCUSDT | 5.42 bp | 100.0% | 0.1 | 218.4K | 60.0% |
| 195 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 66.9% | 0.195 | BTCUSDT | 6.71 bp | 100.0% | 0.1 | 279K | 68.9% |
| 196 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 67.0% | 0.225 | CRMUSDT | 2.38 bp | 100.0% | 0.1 | 69.3K | 42.0% |
| 197 | MAV | MAVUSDT (usdm-futures) | usdm-futures | 65.8% | 0.198 | RESOLVUSDT | 6.73 bp | 100.0% | 0.1 | 166.6K | 70.2% |
| 198 | USUAL | USUALUSDT (usdm-futures) | usdm-futures | 65.6% | 0.256 | BTCUSDT | 6.43 bp | 100.0% | 0.2 | 126.2K | 72.2% |
| 199 | TTWO | TTWOUSDT (usdm-futures) | usdm-futures | 65.6% | 0.159 | HMSTRUSDT | 1.33 bp | 100.0% | 0.1 | 32.7K | 31.1% |
| 200 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 65.2% | 0.204 | BMNRUSDT | 9.93 bp | 100.0% | 0 | 666.5K | 112.6% |
| 201 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 64.6% | 0.215 | BTCUSDT | 12.95 bp | 100.0% | 0 | 2.2M | 153.0% |
| 202 | ONE | ONEUSDT (usdm-futures) | usdm-futures | 64.6% | 0.185 | HEMIUSDT | 10.52 bp | 100.0% | 0.1 | 2.5M | 106.0% |
| 203 | ORCA | ORCAUSDT (usdm-futures) | usdm-futures | 64.8% | 0.303 | BTCUSDT | 3.54 bp | 100.0% | 0.4 | 283.9K | 43.8% |
| 204 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 64.0% | 0.187 | GPSUSDT | 12.72 bp | 100.0% | 0 | 483.5K | 147.7% |
| 205 | ALLO | ALLOUSDT (usdm-futures) | usdm-futures | 64.3% | 0.190 | CLOUSDT | 9.16 bp | 100.0% | 0 | 5.1M | 85.6% |
| 206 | MET | METUSDT (usdm-futures) | usdm-futures | 64.1% | 0.305 | PAYPUSDT | 8.37 bp | 100.0% | 0.1 | 860.3K | 103.3% |
| 207 | PHA | PHAUSDT (usdm-futures) | usdm-futures | 64.2% | 0.290 | BTCUSDT | 6.65 bp | 100.0% | 0.1 | 581.6K | 70.4% |
| 208 | PYTH | PYTHUSDT (usdm-futures) | usdm-futures | 63.7% | 0.282 | BTCUSDT | 5.85 bp | 100.0% | 0.1 | 1.3M | 63.2% |
| 209 | CATI | CATIUSDT (usdm-futures) | usdm-futures | 63.8% | 0.178 | SLPUSDT | 4.48 bp | 100.0% | 0.1 | 93.6K | 43.5% |
| 210 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 63.3% | 0.286 | BUSDT | 10.67 bp | 100.0% | 0 | 724K | 117.1% |
| 211 | PENG | PENGUSDT (usdm-futures) | usdm-futures | 63.1% | 0.183 | XPINUSDT | 8.20 bp | 100.0% | 0 | 503.2K | 101.0% |
| 212 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 62.6% | 0.178 | AGTUSDT | 24.34 bp | 100.0% | 0 | 9.6M | 242.7% |
| 213 | RIVER | RIVERUSDT (usdm-futures) | usdm-futures | 62.5% | 0.318 | BTCUSDT | 6.88 bp | 100.0% | 0.1 | 1.8M | 70.9% |
| 214 | XBI | XBIUSDT (usdm-futures) | usdm-futures | 62.8% | 0.192 | DATAIPUSDT | 1.58 bp | 100.0% | 0.1 | 26.7K | 33.6% |
| 215 | GMX | GMXUSDT (usdm-futures) | usdm-futures | 62.1% | 0.273 | BTCUSDT | 5.03 bp | 100.0% | 0.1 | 285.3K | 55.9% |
| 216 | UB | UBUSDT (usdm-futures) | usdm-futures | 61.7% | 0.234 | AGTUSDT | 15.19 bp | 100.0% | 0 | 4.6M | 147.8% |
| 217 | H | HUSDT (usdm-futures) | usdm-futures | 61.6% | 0.179 | DATAIPUSDT | 5.54 bp | 100.0% | 0.1 | 338.5K | 57.8% |
| 218 | CHEEMS | 1000CHEEMSUSDT (usdm-futures) | usdm-futures | 61.2% | 0.391 | BTCUSDT | 3.97 bp | 100.0% | 0.1 | 165.2K | 42.1% |
| 219 | CC | CCUSDT (usdm-futures) | usdm-futures | 61.3% | 0.188 | XLEUSDT | 3.33 bp | 100.0% | 0 | 555.8K | 37.9% |
| 220 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 60.6% | 0.150 | EDGEUSDT | 10.25 bp | 100.0% | 0 | 351.2K | 98.6% |
| 221 | MUBARAK | MUBARAKUSDT (usdm-futures) | usdm-futures | 60.5% | 0.196 | RECALLUSDT | 5.92 bp | 100.0% | 0.1 | 107K | 64.8% |
| 222 | RED | REDUSDT (usdm-futures) | usdm-futures | 60.4% | 0.232 | BTCUSDT | 5.88 bp | 100.0% | 0.2 | 145.5K | 66.2% |
| 223 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 59.8% | 0.288 | BTCUSDT | 9.85 bp | 100.0% | 0 | 708.2K | 95.9% |
| 224 | DEEP | DEEPUSDT (usdm-futures) | usdm-futures | 59.6% | 0.276 | COLLECTUSDT | 6.43 bp | 100.0% | 0.2 | 407.4K | 88.3% |
| 225 | STX | STXUSDT (usdm-futures) | usdm-futures | 59.7% | 0.288 | BTCUSDT | 4.45 bp | 100.0% | 0.1 | 1.1M | 66.2% |
| 226 | VRT | VRTUSDT (usdm-futures) | usdm-futures | 59.7% | 0.184 | ASMLUSDT | 3.00 bp | 100.0% | 0 | 37.5K | 47.1% |
| 227 | CKB | CKBUSDT (usdm-futures) | usdm-futures | 59.2% | 0.255 | BTCUSDT | 4.90 bp | 100.0% | 0.2 | 130.2K | 60.8% |
| 228 | V | VUSDT (usdm-futures) | usdm-futures | 59.3% | 0.162 | TTWOUSDT | 1.23 bp | 100.0% | 0 | 54.6K | 21.8% |
| 229 | VIC | VICUSDT (usdm-futures) | usdm-futures | 58.8% | 0.212 | EDENUSDT | 4.85 bp | 100.0% | 0.1 | 255.9K | 64.2% |
| 230 | SANTOS | SANTOSUSDT (usdm-futures) | usdm-futures | 58.6% | 0.251 | BTCUSDT | 4.71 bp | 100.0% | 0.1 | 78.6K | 50.1% |
| 231 | XVS | XVSUSDT (usdm-futures) | usdm-futures | 58.5% | 0.271 | BTCUSDT | 3.96 bp | 100.0% | 0.1 | 127.2K | 52.0% |
| 232 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 57.9% | 0.215 | GIGGLEUSDT | 10.31 bp | 100.0% | 0.1 | 2.2M | 104.6% |
| 233 | MEGA | MEGAUSDT (usdm-futures) | usdm-futures | 57.5% | 0.289 | GIGGLEUSDT | 6.34 bp | 100.0% | 0.1 | 589.9K | 65.4% |
| 234 | T | TUSDT (usdm-futures) | usdm-futures | 57.5% | 0.257 | BTCUSDT | 3.48 bp | 100.0% | 0.1 | 847.4K | 37.7% |
| 235 | TLM | TLMUSDT (usdm-futures) | usdm-futures | 56.8% | 0.150 | NMRUSDT | 17.73 bp | 100.0% | 0.1 | 5.2M | 182.3% |
| 236 | MON | MONUSDT (usdm-futures) | usdm-futures | 56.8% | 0.350 | BTCUSDT | 7.54 bp | 100.0% | 0.1 | 2.3M | 78.6% |
| 237 | ORCL | ORCLUSDT (usdm-futures) | usdm-futures | 57.0% | 0.236 | NVDAUSDT | 4.82 bp | 100.0% | 0.1 | 2.7M | 52.8% |

## Diagnostics

- Basis size selected: 237
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.049
- Maximum pairwise absolute correlation: 0.391
- Mean whole-market projection R²: 84.8%
- Median whole-market projection R²: 80.0%
- 10th-percentile whole-market projection R²: 71.3%
- Minimum whole-market projection R²: 68.1%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.6% | 4.2% | 0.1% | 0.0% |
| 5 | 10.2% | 5.6% | 1.1% | 0.1% |
| 10 | 12.3% | 7.1% | 2.5% | 0.4% |
| 15 | 14.4% | 8.8% | 3.8% | 1.0% |
| 20 | 16.9% | 11.0% | 5.4% | 2.3% |
| 25 | 19.3% | 13.2% | 7.0% | 4.1% |
| 30 | 21.3% | 14.7% | 8.6% | 5.3% |
| 35 | 23.4% | 16.4% | 10.1% | 6.9% |
| 40 | 25.5% | 18.2% | 11.6% | 8.0% |
| 45 | 27.8% | 20.5% | 13.0% | 9.3% |
| 50 | 29.7% | 22.1% | 14.4% | 10.6% |
| 55 | 31.7% | 24.1% | 16.1% | 11.9% |
| 60 | 33.8% | 25.7% | 17.7% | 13.5% |
| 65 | 35.7% | 27.6% | 19.4% | 15.6% |
| 70 | 38.2% | 29.8% | 21.1% | 16.6% |
| 75 | 39.9% | 31.0% | 22.6% | 18.9% |
| 80 | 41.6% | 32.6% | 23.9% | 20.2% |
| 85 | 43.6% | 35.1% | 25.8% | 21.4% |
| 90 | 45.5% | 36.5% | 27.3% | 23.2% |
| 95 | 47.3% | 38.3% | 29.0% | 24.4% |
| 100 | 49.1% | 39.8% | 30.4% | 26.5% |
| 105 | 50.7% | 41.4% | 32.1% | 27.9% |
| 110 | 52.2% | 42.7% | 33.7% | 29.4% |
| 115 | 53.9% | 44.6% | 35.3% | 30.7% |
| 120 | 55.5% | 46.4% | 36.7% | 32.4% |
| 125 | 57.2% | 48.2% | 38.2% | 33.9% |
| 130 | 58.8% | 49.8% | 39.9% | 35.6% |
| 135 | 60.3% | 51.1% | 41.4% | 37.9% |
| 140 | 61.8% | 53.2% | 42.9% | 39.4% |
| 145 | 63.2% | 54.4% | 44.6% | 41.2% |
| 150 | 64.6% | 56.2% | 45.9% | 42.4% |
| 155 | 65.9% | 57.3% | 47.5% | 43.4% |
| 160 | 67.3% | 58.8% | 48.9% | 45.3% |
| 165 | 68.5% | 59.9% | 50.4% | 46.5% |
| 170 | 69.8% | 61.3% | 51.7% | 48.5% |
| 175 | 71.1% | 62.7% | 53.4% | 49.8% |
| 180 | 72.4% | 64.3% | 55.1% | 51.4% |
| 185 | 73.7% | 66.1% | 56.5% | 52.9% |
| 190 | 75.0% | 67.7% | 58.1% | 53.9% |
| 195 | 76.1% | 69.3% | 59.4% | 55.1% |
| 200 | 77.2% | 70.8% | 61.1% | 57.8% |
| 205 | 78.3% | 71.9% | 62.3% | 58.7% |
| 210 | 79.4% | 73.2% | 63.8% | 60.2% |
| 215 | 80.5% | 74.4% | 65.4% | 61.8% |
| 220 | 81.5% | 75.7% | 66.6% | 63.3% |
| 225 | 82.5% | 77.1% | 67.9% | 64.4% |
| 230 | 83.4% | 78.3% | 69.4% | 65.8% |
| 235 | 84.4% | 79.4% | 70.7% | 67.5% |
| 237 | 84.8% | 80.0% | 71.3% | 68.1% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | B2 | RIF | AKE | ON | DEXE | BLESS | AIA | AGT | RE | GWEI | XNY | ANTHROPIC | MINIMAX | SPORTFUN | BEL | HPE | BNC | LLY | DIS | US | COST | DODOX | BULLA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | -0.060 | -0.055 | 0.028 | 0.039 | 0.017 | -0.007 | 0.014 | 0.011 | 0.034 | -0.013 | 0.022 | -0.028 | 0.029 | 0.037 | -0.015 | 0.002 | 0.053 | 0.007 | -0.005 | 0.107 | 0.032 | 0.025 | 0.002 |
| B2 | -0.060 | 1.000 | 0.027 | -0.046 | -0.055 | -0.008 | -0.015 | 0.006 | 0.020 | 0.052 | -0.017 | 0.032 | 0.012 | 0.008 | 0.015 | 0.059 | -0.031 | -0.015 | 0.020 | -0.026 | -0.040 | -0.018 | -0.024 | -0.048 |
| RIF | -0.055 | 0.027 | 1.000 | -0.037 | -0.037 | 0.003 | -0.002 | 0.052 | 0.023 | -0.016 | 0.013 | -0.007 | -0.015 | 0.008 | 0.056 | -0.017 | 0.057 | 0.022 | 0.020 | -0.042 | -0.047 | -0.025 | -0.014 | -0.028 |
| AKE | 0.028 | -0.046 | -0.037 | 1.000 | -0.042 | -0.001 | 0.070 | 0.003 | 0.000 | -0.012 | -0.002 | -0.002 | -0.003 | 0.023 | 0.032 | -0.051 | -0.021 | 0.002 | -0.007 | 0.049 | 0.037 | 0.015 | 0.032 | -0.034 |
| ON | 0.039 | -0.055 | -0.037 | -0.042 | 1.000 | 0.021 | -0.004 | -0.019 | -0.051 | -0.004 | 0.024 | -0.008 | -0.004 | -0.027 | 0.039 | -0.046 | 0.029 | -0.035 | -0.000 | -0.085 | 0.001 | 0.050 | 0.023 | -0.076 |
| DEXE | 0.017 | -0.008 | 0.003 | -0.001 | 0.021 | 1.000 | 0.007 | -0.003 | 0.034 | 0.021 | -0.057 | -0.025 | -0.017 | 0.008 | -0.050 | 0.004 | 0.005 | -0.010 | -0.031 | -0.013 | -0.091 | 0.076 | 0.076 | -0.047 |
| BLESS | -0.007 | -0.015 | -0.002 | 0.070 | -0.004 | 0.007 | 1.000 | -0.049 | -0.034 | 0.075 | 0.002 | 0.045 | 0.001 | -0.017 | 0.008 | 0.033 | 0.000 | -0.036 | 0.043 | -0.031 | -0.051 | 0.001 | 0.069 | -0.015 |
| AIA | 0.014 | 0.006 | 0.052 | 0.003 | -0.019 | -0.003 | -0.049 | 1.000 | 0.068 | 0.004 | -0.061 | -0.005 | -0.012 | 0.017 | -0.032 | -0.018 | -0.007 | -0.014 | -0.053 | -0.044 | -0.028 | -0.021 | -0.033 | -0.072 |
| AGT | 0.011 | 0.020 | 0.023 | 0.000 | -0.051 | 0.034 | -0.034 | 0.068 | 1.000 | 0.001 | -0.032 | 0.009 | -0.064 | 0.015 | -0.018 | 0.033 | -0.063 | 0.014 | 0.004 | 0.046 | 0.035 | -0.032 | 0.068 | 0.001 |
| RE | 0.034 | 0.052 | -0.016 | -0.012 | -0.004 | 0.021 | 0.075 | 0.004 | 0.001 | 1.000 | 0.022 | -0.031 | -0.046 | -0.024 | -0.007 | -0.012 | 0.004 | -0.016 | -0.015 | -0.034 | -0.005 | 0.026 | 0.024 | -0.048 |
| GWEI | -0.013 | -0.017 | 0.013 | -0.002 | 0.024 | -0.057 | 0.002 | -0.061 | -0.032 | 0.022 | 1.000 | 0.031 | 0.019 | -0.012 | -0.046 | -0.030 | 0.010 | 0.031 | 0.035 | -0.001 | -0.020 | 0.013 | 0.066 | 0.004 |
| XNY | 0.022 | 0.032 | -0.007 | -0.002 | -0.008 | -0.025 | 0.045 | -0.005 | 0.009 | -0.031 | 0.031 | 1.000 | 0.021 | 0.035 | -0.004 | 0.034 | -0.026 | 0.006 | 0.046 | -0.022 | 0.001 | -0.019 | 0.043 | 0.067 |
| ANTHROPIC | -0.028 | 0.012 | -0.015 | -0.003 | -0.004 | -0.017 | 0.001 | -0.012 | -0.064 | -0.046 | 0.019 | 0.021 | 1.000 | 0.017 | 0.004 | 0.003 | -0.043 | -0.021 | -0.014 | 0.052 | -0.004 | 0.024 | -0.033 | 0.048 |
| MINIMAX | 0.029 | 0.008 | 0.008 | 0.023 | -0.027 | 0.008 | -0.017 | 0.017 | 0.015 | -0.024 | -0.012 | 0.035 | 0.017 | 1.000 | 0.009 | -0.015 | 0.018 | 0.073 | -0.014 | -0.046 | 0.011 | 0.032 | -0.029 | 0.033 |
| SPORTFUN | 0.037 | 0.015 | 0.056 | 0.032 | 0.039 | -0.050 | 0.008 | -0.032 | -0.018 | -0.007 | -0.046 | -0.004 | 0.004 | 0.009 | 1.000 | -0.045 | 0.057 | -0.063 | -0.001 | -0.000 | -0.004 | 0.031 | 0.069 | -0.067 |
| BEL | -0.015 | 0.059 | -0.017 | -0.051 | -0.046 | 0.004 | 0.033 | -0.018 | 0.033 | -0.012 | -0.030 | 0.034 | 0.003 | -0.015 | -0.045 | 1.000 | 0.012 | -0.043 | 0.063 | 0.004 | 0.035 | -0.023 | -0.019 | -0.002 |
| HPE | 0.002 | -0.031 | 0.057 | -0.021 | 0.029 | 0.005 | 0.000 | -0.007 | -0.063 | 0.004 | 0.010 | -0.026 | -0.043 | 0.018 | 0.057 | 0.012 | 1.000 | 0.008 | -0.031 | -0.041 | -0.004 | -0.046 | -0.045 | -0.032 |
| BNC | 0.053 | -0.015 | 0.022 | 0.002 | -0.035 | -0.010 | -0.036 | -0.014 | 0.014 | -0.016 | 0.031 | 0.006 | -0.021 | 0.073 | -0.063 | -0.043 | 0.008 | 1.000 | 0.006 | 0.063 | 0.012 | -0.015 | -0.029 | -0.055 |
| LLY | 0.007 | 0.020 | 0.020 | -0.007 | -0.000 | -0.031 | 0.043 | -0.053 | 0.004 | -0.015 | 0.035 | 0.046 | -0.014 | -0.014 | -0.001 | 0.063 | -0.031 | 0.006 | 1.000 | -0.013 | -0.022 | -0.050 | -0.020 | 0.023 |
| DIS | -0.005 | -0.026 | -0.042 | 0.049 | -0.085 | -0.013 | -0.031 | -0.044 | 0.046 | -0.034 | -0.001 | -0.022 | 0.052 | -0.046 | -0.000 | 0.004 | -0.041 | 0.063 | -0.013 | 1.000 | 0.010 | -0.050 | 0.010 | 0.036 |
| US | 0.107 | -0.040 | -0.047 | 0.037 | 0.001 | -0.091 | -0.051 | -0.028 | 0.035 | -0.005 | -0.020 | 0.001 | -0.004 | 0.011 | -0.004 | 0.035 | -0.004 | 0.012 | -0.022 | 0.010 | 1.000 | -0.024 | 0.023 | -0.032 |
| COST | 0.032 | -0.018 | -0.025 | 0.015 | 0.050 | 0.076 | 0.001 | -0.021 | -0.032 | 0.026 | 0.013 | -0.019 | 0.024 | 0.032 | 0.031 | -0.023 | -0.046 | -0.015 | -0.050 | -0.050 | -0.024 | 1.000 | 0.050 | 0.020 |
| DODOX | 0.025 | -0.024 | -0.014 | 0.032 | 0.023 | 0.076 | 0.069 | -0.033 | 0.068 | 0.024 | 0.066 | 0.043 | -0.033 | -0.029 | 0.069 | -0.019 | -0.045 | -0.029 | -0.020 | 0.010 | 0.023 | 0.050 | 1.000 | 0.050 |
| BULLA | 0.002 | -0.048 | -0.028 | -0.034 | -0.076 | -0.047 | -0.015 | -0.072 | 0.001 | -0.048 | 0.004 | 0.067 | 0.048 | 0.033 | -0.067 | -0.002 | -0.032 | -0.055 | 0.023 | 0.036 | -0.032 | 0.020 | 0.050 | 1.000 |

The complete 237 × 237 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| TRX | TRXUSDT | 68.1% | 56.4% | BTCUSDT | 0.234 |
| HEI | HEIUSDT | 68.2% | 56.4% | TAIKOUSDT | -0.164 |
| QNTX | QNTXUSDT | 68.3% | 56.3% | BROCCOLIF3BUSDT | -0.194 |
| PLUME | PLUMEUSDT | 68.6% | 56.0% | BTCUSDT | 0.230 |
| COW | COWUSDT | 68.6% | 56.0% | BTCUSDT | 0.338 |
| KAVA | KAVAUSDT | 68.7% | 55.9% | DISUSDT | 0.205 |
| SCR | SCRUSDT | 68.8% | 55.8% | BTCUSDT | 0.230 |
| CFG | CFGUSDT | 68.9% | 55.8% | BTCUSDT | 0.242 |
| TZA | TZAUSDT | 68.9% | 55.7% | SPKUSDT | -0.182 |
| SNOW | SNOWUSDT | 69.0% | 55.7% | CRMUSDT | 0.247 |
| BNT | BNTUSDT | 69.0% | 55.7% | XVSUSDT | 0.340 |
| IWM | IWMUSDT | 69.0% | 55.7% | SONYUSDT | 0.246 |
| SATS | 1000SATSUSDT | 69.0% | 55.7% | HOMEUSDT | 0.209 |
| JST | JSTUSDT | 69.0% | 55.7% | BASUSDT | 0.202 |
| VANRY | VANRYUSDT | 69.1% | 55.6% | TUSDT | 0.171 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

