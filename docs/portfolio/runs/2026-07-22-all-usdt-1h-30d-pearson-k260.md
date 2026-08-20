# Binance portfolio basis

Generated 2026-07-23T18:10:09.380Z.

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
- Return universe: 580 eligible assets from 715 active continuously priced candidates
- Quality filter: at least 80.0% non-zero 1h returns and a complete window
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Median 1h quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 33.3M | 42.5% |
| 2 | ATM | ATMUSDT (spot) | spot | 100.0% | 0.005 | BTCUSDT | 40.7K | 238.2% |
| 3 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 100.0% | 0.015 | BTCUSDT | 122.2K | 421.1% |
| 4 | HD | HDUSDT (usdm-futures) | usdm-futures | 100.0% | 0.015 | BTCUSDT | 5.1K | 32.8% |
| 5 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 100.0% | 0.016 | BTCUSDT | 214.9K | 339.3% |
| 6 | US | USUSDT (usdm-futures) | usdm-futures | 99.9% | 0.025 | DEXEUSDT | 499.5K | 271.5% |
| 7 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 99.9% | 0.028 | USUSDT | 372.7K | 303.8% |
| 8 | HEI | HEIUSDT (spot) | spot, usdm-futures | 99.8% | 0.036 | ATMUSDT | 115.9K | 246.4% |
| 9 | XNO | XNOUSDT (spot) | spot | 99.7% | 0.045 | DEXEUSDT | 1.2K | 233.9% |
| 10 | ADBE | ADBEUSDT (usdm-futures) | usdm-futures | 99.6% | 0.042 | TAIKOUSDT | 14.7K | 48.1% |
| 11 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 99.6% | 0.046 | BTCUSDT | 877K | 690.6% |
| 12 | PIVX | PIVXUSDT (spot) | spot | 99.5% | 0.056 | BTCUSDT | 10.3K | 280.2% |
| 13 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 99.5% | 0.090 | BTCUSDT | 36.5K | 192.3% |
| 14 | BAS | BASUSDT (usdm-futures) | usdm-futures | 99.4% | 0.066 | ATMUSDT | 271.4K | 327.8% |
| 15 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.3% | 0.064 | BTCUSDT | 108.4K | 378.2% |
| 16 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 99.2% | 0.052 | BTCUSDT | 110K | 139.5% |
| 17 | DODO | DODOUSDT (spot) | spot | 99.2% | 0.106 | BTCUSDT | 23.5K | 246.2% |
| 18 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 99.1% | 0.080 | BTCUSDT | 30.7K | 133.8% |
| 19 | M | MUSDT (usdm-futures) | usdm-futures | 99.0% | 0.074 | BTCUSDT | 298.5K | 721.8% |
| 20 | B | BUSDT (usdm-futures) | usdm-futures | 99.0% | 0.095 | EVAAUSDT | 153.7K | 299.5% |
| 21 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 98.9% | 0.116 | BTCUSDT | 26.3K | 219.7% |
| 22 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 98.8% | 0.058 | ATMUSDT | 53.4K | 262.0% |
| 23 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 98.7% | 0.113 | BTCUSDT | 98.3K | 148.8% |
| 24 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 98.6% | 0.080 | ADBEUSDT | 391.9K | 42.5% |
| 25 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 98.5% | 0.114 | BTCUSDT | 25.3K | 102.3% |
| 26 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 98.4% | 0.077 | HANAUSDT | 93.4K | 178.6% |
| 27 | B2 | B2USDT (usdm-futures) | usdm-futures | 98.3% | 0.104 | BTCUSDT | 27.9K | 106.9% |
| 28 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 98.3% | 0.142 | BTCUSDT | 494.5K | 230.4% |
| 29 | CATI | CATIUSDT (spot) | spot, usdm-futures | 98.2% | 0.070 | NATGASUSDT | 11.7K | 104.3% |
| 30 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 98.1% | 0.099 | HEIUSDT | 91K | 222.0% |
| 31 | TLM | TLMUSDT (spot) | spot, usdm-futures | 98.0% | 0.107 | TAIKOUSDT | 97.5K | 408.1% |
| 32 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 97.7% | 0.134 | BTCUSDT | 211.1K | 205.8% |
| 33 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 97.7% | 0.100 | BTCUSDT | 84.2K | 95.6% |
| 34 | BR | BRUSDT (usdm-futures) | usdm-futures | 97.6% | 0.096 | MAGMAUSDT | 48.1K | 139.3% |
| 35 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 97.4% | 0.099 | EDGEUSDT | 47.6K | 116.6% |
| 36 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 97.2% | 0.117 | HDUSDT | 7.8K | 24.9% |
| 37 | G | GUSDT (spot) | spot, usdm-futures | 97.1% | 0.153 | BTCUSDT | 19.8K | 178.8% |
| 38 | UB | UBUSDT (usdm-futures) | usdm-futures | 97.1% | 0.113 | USUSDT | 736K | 261.3% |
| 39 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 97.0% | 0.102 | MANTAUSDT | 688.1K | 316.2% |
| 40 | GPS | GPSUSDT (spot) | spot, usdm-futures | 96.9% | 0.126 | EPICUSDT | 11.9K | 104.3% |
| 41 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 96.8% | 0.102 | PIVXUSDT | 1.3M | 273.4% |
| 42 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 96.7% | 0.153 | BTCUSDT | 57.3K | 166.1% |
| 43 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 96.6% | 0.125 | BTCUSDT | 59.2K | 171.0% |
| 44 | APR | APRUSDT (usdm-futures) | usdm-futures | 96.6% | 0.174 | BTCUSDT | 91K | 121.3% |
| 45 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 96.3% | 0.138 | BTCUSDT | 96.2K | 173.4% |
| 46 | TA | TAUSDT (usdm-futures) | usdm-futures | 96.1% | 0.117 | AGTUSDT | 47.9K | 111.8% |
| 47 | SYN | SYNUSDT (spot) | spot, usdm-futures | 96.0% | 0.138 | HEIUSDT | 445.7K | 373.7% |
| 48 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 95.9% | 0.125 | MUSDT | 483.1K | 264.9% |
| 49 | AIN | AINUSDT (usdm-futures) | usdm-futures | 95.8% | 0.157 | BTCUSDT | 42.3K | 167.3% |
| 50 | PORTO | PORTOUSDT (spot) | spot | 95.6% | 0.177 | BTCUSDT | 7.7K | 183.4% |
| 51 | ON | ONUSDT (usdm-futures) | usdm-futures | 95.5% | 0.143 | BTCUSDT | 40.8K | 200.4% |
| 52 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 95.4% | 0.122 | XPINUSDT | 340.8K | 283.3% |
| 53 | DKNG | DKNGUSDT (usdm-futures) | usdm-futures | 95.3% | 0.126 | ADBEUSDT | 3.6K | 56.9% |
| 54 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 95.2% | 0.157 | BTCUSDT | 1.9M | 268.4% |
| 55 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 95.0% | 0.122 | BTCUSDT | 660.4K | 428.3% |
| 56 | PROM | PROMUSDT (spot) | spot, usdm-futures | 94.9% | 0.156 | DEXEUSDT | 7.1K | 157.3% |
| 57 | HOT | HOTUSDT (spot) | spot, usdm-futures | 94.9% | 0.226 | BTCUSDT | 8.8K | 154.1% |
| 58 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 94.8% | 0.141 | SKYAIUSDT | 81.5K | 203.4% |
| 59 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 94.8% | 0.153 | BTCUSDT | 25.2K | 188.0% |
| 60 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 94.6% | 0.150 | TAIKOUSDT | 83.6K | 186.1% |
| 61 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 94.5% | 0.195 | BTCUSDT | 3.1K | 33.0% |
| 62 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 94.4% | 0.185 | BTCUSDT | 65.5K | 140.6% |
| 63 | PUNDIX | PUNDIXUSDT (spot) | spot, usdm-futures | 94.4% | 0.177 | BTCUSDT | 8.3K | 149.0% |
| 64 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 94.0% | 0.132 | ALCHUSDT | 437.4K | 248.4% |
| 65 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 93.9% | 0.153 | USUSDT | 89.4K | 190.1% |
| 66 | ONG | ONGUSDT (spot) | spot, usdm-futures | 93.9% | 0.236 | BTCUSDT | 6.1K | 121.7% |
| 67 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 93.8% | 0.201 | BTCUSDT | 30.6K | 142.4% |
| 68 | XLE | XLEUSDT (usdm-futures) | usdm-futures | 93.6% | 0.136 | EDGEUSDT | 3.3K | 27.2% |
| 69 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 93.5% | 0.203 | BTCUSDT | 45.2K | 56.1% |
| 70 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 93.4% | 0.146 | BTCUSDT | 53.1K | 166.3% |
| 71 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 93.2% | 0.244 | BTCUSDT | 7.1K | 157.7% |
| 72 | TAC | TACUSDT (usdm-futures) | usdm-futures | 93.1% | 0.184 | ADBEUSDT | 431.4K | 854.7% |
| 73 | ACT | ACTUSDT (spot) | spot, usdm-futures | 92.9% | 0.246 | BTCUSDT | 23.2K | 201.9% |
| 74 | STAR | STARUSDT (usdm-futures) | usdm-futures | 92.7% | 0.188 | BTCUSDT | 49.6K | 186.5% |
| 75 | CRWD | CRWDUSDT (usdm-futures) | usdm-futures | 92.7% | 0.192 | ALLOUSDT | 17.3K | 478.7% |
| 76 | 4 | 4USDT (usdm-futures) | usdm-futures | 92.6% | 0.227 | BTCUSDT | 74.1K | 146.2% |
| 77 | BEL | BELUSDT (spot) | spot, usdm-futures | 92.4% | 0.205 | BTCUSDT | 44.1K | 184.9% |
| 78 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 92.4% | 0.183 | BTCUSDT | 52.6K | 154.2% |
| 79 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 92.3% | 0.171 | TAIKOUSDT | 1.7M | 396.0% |
| 80 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 92.2% | 0.253 | BTCUSDT | 52.7K | 125.0% |
| 81 | OGN | OGNUSDT (spot) | spot, usdm-futures | 92.1% | 0.266 | BTCUSDT | 17.3K | 160.0% |
| 82 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 92.1% | 0.158 | EPICUSDT | 81.7K | 289.2% |
| 83 | SENT | SENTUSDT (spot) | spot, usdm-futures | 91.9% | 0.251 | BTCUSDT | 50.3K | 116.3% |
| 84 | GNO | GNOUSDT (spot) | spot | 91.8% | 0.255 | BTCUSDT | 3.2K | 65.7% |
| 85 | MMT | MMTUSDT (spot) | spot, usdm-futures | 91.6% | 0.146 | BTCUSDT | 35K | 141.5% |
| 86 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 91.6% | 0.265 | ADBEUSDT | 21.4K | 47.9% |
| 87 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 91.5% | 0.202 | BTCUSDT | 286.7K | 229.2% |
| 88 | FF | FFUSDT (spot) | spot, usdm-futures | 91.5% | 0.169 | BTCUSDT | 39.3K | 73.1% |
| 89 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 91.3% | 0.174 | BTCUSDT | 39.3K | 99.9% |
| 90 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 91.2% | 0.178 | BTCUSDT | 46.1K | 164.5% |
| 91 | QUICK | QUICKUSDT (spot) | spot | 91.1% | 0.194 | BRUSDT | 2.4K | 163.4% |
| 92 | ID | IDUSDT (spot) | spot, usdm-futures | 91.0% | 0.192 | BTCUSDT | 45.1K | 137.1% |
| 93 | JST | JSTUSDT (spot) | spot, usdm-futures | 90.9% | 0.165 | BTCUSDT | 98K | 43.6% |
| 94 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 90.8% | 0.169 | ESPORTSUSDT | 49.9K | 118.6% |
| 95 | Q | QUSDT (usdm-futures) | usdm-futures | 90.5% | 0.154 | BTCUSDT | 38.7K | 102.5% |
| 96 | SKL | SKLUSDT (spot) | spot, usdm-futures | 90.5% | 0.285 | BTCUSDT | 15K | 153.1% |
| 97 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 90.4% | 0.228 | BTCUSDT | 45.9K | 199.7% |
| 98 | XEC | XECUSDT (spot) | spot, usdm-futures | 90.3% | 0.206 | BTCUSDT | 9.7K | 164.7% |
| 99 | THE | THEUSDT (spot) | spot, usdm-futures | 90.3% | 0.218 | BTCUSDT | 18.5K | 207.4% |
| 100 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 90.2% | 0.158 | DKNGUSDT | 6.8K | 35.5% |
| 101 | IN | INUSDT (usdm-futures) | usdm-futures | 90.1% | 0.145 | BULLAUSDT | 201.8K | 300.4% |
| 102 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 89.9% | 0.190 | BTCUSDT | 45.6K | 112.6% |
| 103 | OPG | OPGUSDT (spot) | spot, usdm-futures | 89.8% | 0.227 | BTCUSDT | 58.1K | 175.2% |
| 104 | BAN | BANUSDT (usdm-futures) | usdm-futures | 89.8% | 0.205 | BTCUSDT | 32.4K | 99.2% |
| 105 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 89.7% | 0.240 | BTCUSDT | 161.9K | 162.3% |
| 106 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 89.6% | 0.249 | BTCUSDT | 80K | 155.8% |
| 107 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 89.4% | 0.246 | BTCUSDT | 53.3K | 121.4% |
| 108 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 89.3% | 0.167 | BTCUSDT | 70.7K | 148.1% |
| 109 | ACE | ACEUSDT (spot) | spot, usdm-futures | 89.2% | 0.229 | BTCUSDT | 10.1K | 189.4% |
| 110 | AWE | AWEUSDT (spot) | spot, usdm-futures | 89.0% | 0.174 | BTCUSDT | 9.8K | 108.3% |
| 111 | RIF | RIFUSDT (spot) | spot, usdm-futures | 88.9% | 0.183 | DEXEUSDT | 68.2K | 252.9% |
| 112 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 88.8% | 0.205 | BTCUSDT | 103.1K | 187.0% |
| 113 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 88.8% | 0.136 | XECUSDT | 116.7K | 182.9% |
| 114 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 88.5% | 0.173 | ACTUSDT | 129.7K | 159.2% |
| 115 | LAB | LABUSDT (usdm-futures) | usdm-futures | 88.3% | 0.193 | TACUSDT | 13.8M | 521.9% |
| 116 | SXT | SXTUSDT (spot) | spot, usdm-futures | 88.3% | 0.237 | BTCUSDT | 24.5K | 156.2% |
| 117 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 88.2% | 0.183 | BTCUSDT | 81.8K | 219.6% |
| 118 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 88.1% | 0.167 | MMTUSDT | 33K | 195.7% |
| 119 | BANK | BANKUSDT (spot) | spot, usdm-futures | 87.9% | 0.257 | DEXEUSDT | 14.7K | 432.1% |
| 120 | ERA | ERAUSDT (spot) | spot, usdm-futures | 87.8% | 0.292 | BTCUSDT | 7.8K | 174.8% |
| 121 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 87.8% | 0.174 | BTCUSDT | 27.1K | 126.6% |
| 122 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 87.6% | 0.154 | BTCUSDT | 56.3K | 199.3% |
| 123 | XAN | XANUSDT (usdm-futures) | usdm-futures | 87.4% | 0.242 | BTCUSDT | 88.8K | 164.4% |
| 124 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 87.3% | 0.235 | BTCUSDT | 15.6K | 87.7% |
| 125 | QKC | QKCUSDT (spot) | spot | 87.2% | 0.223 | PHAROSUSDT | 3.3K | 144.8% |
| 126 | AT | ATUSDT (spot) | spot, usdm-futures | 87.1% | 0.202 | BTCUSDT | 7.8K | 88.6% |
| 127 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 86.8% | 0.206 | BTCUSDT | 650.1K | 205.5% |
| 128 | IQ | IQUSDT (spot) | spot | 86.7% | 0.253 | BTCUSDT | 1.2K | 74.5% |
| 129 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 86.6% | 0.293 | BTCUSDT | 9.8K | 134.2% |
| 130 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 86.5% | 0.278 | BTCUSDT | 423.3K | 71.8% |
| 131 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 86.2% | 0.151 | BTCUSDT | 34.9K | 142.6% |
| 132 | SUN | SUNUSDT (spot) | spot, usdm-futures | 86.1% | 0.291 | BTCUSDT | 18.8K | 25.8% |
| 133 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 86.0% | 0.194 | GPSUSDT | 95.5K | 217.1% |
| 134 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 85.9% | 0.229 | BTCUSDT | 351.8K | 222.3% |
| 135 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 85.6% | 0.218 | BTCUSDT | 28.5K | 122.5% |
| 136 | CC | CCUSDT (usdm-futures) | usdm-futures | 85.5% | 0.222 | BTCUSDT | 144.1K | 61.2% |
| 137 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 85.2% | 0.191 | UBUSDT | 61.8K | 214.2% |
| 138 | GME | GMEUSDT (usdm-futures) | usdm-futures | 85.0% | 0.210 | BTCUSDT | 2.8K | 31.5% |
| 139 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 84.8% | 0.276 | BTCUSDT | 13.8K | 154.2% |
| 140 | KITE | KITEUSDT (spot) | spot, usdm-futures | 84.8% | 0.286 | BTCUSDT | 143.2K | 121.3% |
| 141 | CITY | CITYUSDT (spot) | spot | 84.6% | 0.221 | BTCUSDT | 14.8K | 119.8% |
| 142 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 84.3% | 0.214 | BTCUSDT | 117.7K | 154.3% |
| 143 | T | TUSDT (spot) | spot, usdm-futures | 84.3% | 0.289 | BTCUSDT | 15.7K | 153.7% |
| 144 | ZKP | ZKPUSDT (spot) | spot, usdm-futures | 84.1% | 0.271 | BTCUSDT | 11.1K | 131.2% |
| 145 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 84.0% | 0.327 | BTCUSDT | 34.1K | 128.7% |
| 146 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 83.9% | 0.281 | BTCUSDT | 35.6K | 117.3% |
| 147 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 83.9% | 0.274 | BTCUSDT | 24.6K | 97.3% |
| 148 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 83.6% | 0.290 | BTCUSDT | 22K | 153.3% |
| 149 | RE | REUSDT (spot) | spot, usdm-futures | 83.5% | 0.257 | BTCUSDT | 607.6K | 180.3% |
| 150 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 83.4% | 0.304 | BTCUSDT | 58.5K | 157.7% |
| 151 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 83.1% | 0.301 | BTCUSDT | 16.2K | 68.5% |
| 152 | BX | BXUSDT (usdm-futures) | usdm-futures | 83.1% | 0.178 | BTCUSDT | 11.1K | 47.6% |
| 153 | FTT | FTTUSDT (spot) | spot | 82.9% | 0.320 | BTCUSDT | 6.9K | 122.2% |
| 154 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 82.9% | 0.327 | BTCUSDT | 42.5K | 120.1% |
| 155 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 82.8% | 0.196 | BTCUSDT | 438.8K | 191.7% |
| 156 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 82.6% | 0.245 | EDGEUSDT | 89.5K | 327.3% |
| 157 | DCR | DCRUSDT (spot) | spot | 82.5% | 0.274 | BTCUSDT | 7.4K | 128.0% |
| 158 | POWR | POWRUSDT (spot) | spot, usdm-futures | 82.4% | 0.326 | BTCUSDT | 4.8K | 130.4% |
| 159 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 82.3% | 0.217 | BTCUSDT | 53.1K | 96.6% |
| 160 | STRAX | STRAXUSDT (spot) | spot | 82.1% | 0.256 | BTCUSDT | 13.6K | 126.7% |
| 161 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 82.1% | 0.246 | BROCCOLIF3BUSDT | 157.8K | 243.5% |
| 162 | V | VUSDT (usdm-futures) | usdm-futures | 81.9% | 0.210 | HDUSDT | 4.5K | 30.9% |
| 163 | TUT | TUTUSDT (spot) | spot, usdm-futures | 81.8% | 0.285 | BTCUSDT | 11.7K | 103.3% |
| 164 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 81.4% | 0.271 | BTCUSDT | 75.9K | 85.8% |
| 165 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 81.2% | 0.354 | BTCUSDT | 22.9K | 83.4% |
| 166 | AUDIO | AUDIOUSDT (spot) | spot | 81.1% | 0.283 | BTCUSDT | 8.1K | 105.0% |
| 167 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 80.9% | 0.291 | BTCUSDT | 37.2K | 136.6% |
| 168 | REQ | REQUSDT (spot) | spot | 80.6% | 0.401 | BTCUSDT | 2.5K | 69.5% |
| 169 | JTO | JTOUSDT (spot) | spot, usdm-futures | 80.5% | 0.285 | BTCUSDT | 155.8K | 129.7% |
| 170 | MBL | MBLUSDT (spot) | spot | 80.3% | 0.255 | QKCUSDT | 22.3K | 97.9% |
| 171 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 80.2% | 0.273 | BTCUSDT | 97.2K | 187.2% |
| 172 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 80.0% | 0.440 | BTCUSDT | 32.5K | 123.2% |
| 173 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 79.9% | 0.238 | CRWDUSDT | 6.3K | 26.6% |
| 174 | GUN | GUNUSDT (spot) | spot, usdm-futures | 79.9% | 0.380 | BTCUSDT | 20.5K | 125.7% |
| 175 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 79.7% | 0.371 | BTCUSDT | 19.9K | 106.2% |
| 176 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 79.6% | 0.332 | BTCUSDT | 45.5K | 137.8% |
| 177 | NMR | NMRUSDT (spot) | spot, usdm-futures | 79.4% | 0.372 | BTCUSDT | 10.4K | 74.7% |
| 178 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 79.3% | 0.201 | GMEUSDT | 3.5K | 33.6% |
| 179 | LIT | LITUSDT (usdm-futures) | usdm-futures | 79.2% | 0.378 | BTCUSDT | 1.8M | 138.7% |
| 180 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 79.0% | 0.375 | BTCUSDT | 91.1K | 127.1% |
| 181 | COPPER | COPPERUSDT (usdm-futures) | usdm-futures | 78.9% | 0.382 | BTCUSDT | 120.5K | 23.8% |
| 182 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 78.5% | 0.365 | BTCUSDT | 67.5K | 97.1% |
| 183 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 78.3% | 0.380 | BTCUSDT | 40.4K | 128.8% |
| 184 | CELO | CELOUSDT (spot) | spot, usdm-futures | 78.2% | 0.350 | BTCUSDT | 30.6K | 115.4% |
| 185 | STG | STGUSDT (spot) | spot, usdm-futures | 78.1% | 0.278 | BTCUSDT | 26.9K | 120.7% |
| 186 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 77.9% | 0.229 | IDOLUSDT | 23.8K | 79.1% |
| 187 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 77.6% | 0.248 | CRWDUSDT | 240.6K | 315.8% |
| 188 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 77.6% | 0.323 | BTCUSDT | 25.3K | 104.3% |
| 189 | TREE | TREEUSDT (spot) | spot, usdm-futures | 77.2% | 0.366 | BTCUSDT | 22K | 129.1% |
| 190 | C | CUSDT (spot) | spot, usdm-futures | 77.1% | 0.384 | BTCUSDT | 8.7K | 86.6% |
| 191 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 76.9% | 0.202 | HDUSDT | 6.4K | 41.4% |
| 192 | XPL | XPLUSDT (spot) | spot, usdm-futures | 76.8% | 0.424 | BTCUSDT | 302K | 125.7% |
| 193 | SKY | SKYUSDT (spot) | spot, usdm-futures | 76.6% | 0.459 | BTCUSDT | 38.4K | 74.8% |
| 194 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 76.4% | 0.272 | BTCUSDT | 7.4K | 75.1% |
| 195 | HYPER | HYPERUSDT (spot) | spot, usdm-futures | 76.2% | 0.390 | BTCUSDT | 34.9K | 88.4% |
| 196 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 76.1% | 0.384 | BTCUSDT | 23K | 91.8% |
| 197 | TST | TSTUSDT (spot) | spot, usdm-futures | 75.5% | 0.307 | BTCUSDT | 17.1K | 111.4% |
| 198 | ELSA | ELSAUSDT (usdm-futures) | usdm-futures | 75.4% | 0.324 | BTCUSDT | 51.9K | 111.6% |
| 199 | H | HUSDT (usdm-futures) | usdm-futures | 75.2% | 0.323 | MUSDT | 419.3K | 281.4% |
| 200 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 75.2% | 0.314 | BTCUSDT | 233.8K | 195.1% |
| 201 | ARK | ARKUSDT (spot) | spot, usdm-futures | 75.1% | 0.401 | BTCUSDT | 1.9K | 116.9% |
| 202 | GRASS | GRASSUSDT (usdm-futures) | usdm-futures | 75.0% | 0.384 | BTCUSDT | 297.7K | 155.2% |
| 203 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 74.8% | 0.293 | BXUSDT | 6.5K | 28.8% |
| 204 | HEMI | HEMIUSDT (spot) | spot, usdm-futures | 74.7% | 0.282 | TOWNSUSDT | 19.8K | 159.9% |
| 205 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 74.5% | 0.375 | BTCUSDT | 38.3K | 110.8% |
| 206 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 74.4% | 0.445 | BTCUSDT | 15.7K | 99.3% |
| 207 | OPN | OPNUSDT (spot) | spot, usdm-futures | 74.3% | 0.314 | BTCUSDT | 119.3K | 128.8% |
| 208 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 74.2% | 0.315 | BTCUSDT | 28.7K | 115.0% |
| 209 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 74.0% | 0.369 | BTCUSDT | 21.8K | 101.2% |
| 210 | MITO | MITOUSDT (spot) | spot, usdm-futures | 73.8% | 0.290 | BTCUSDT | 19.6K | 121.6% |
| 211 | POPCAT | POPCATUSDT (usdm-futures) | usdm-futures | 73.7% | 0.450 | BTCUSDT | 78.4K | 121.1% |
| 212 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 73.6% | 0.399 | BTCUSDT | 115.5K | 109.6% |
| 213 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 73.6% | 0.341 | BTCUSDT | 427.9K | 218.2% |
| 214 | WIN | WINUSDT (spot) | spot | 73.5% | 0.468 | BTCUSDT | 3K | 54.4% |
| 215 | MET | METUSDT (spot) | spot, usdm-futures | 73.4% | 0.382 | BTCUSDT | 29K | 134.2% |
| 216 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 73.3% | 0.357 | BTCUSDT | 24.3K | 220.6% |
| 217 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 73.2% | 0.402 | BTCUSDT | 7.1K | 80.2% |
| 218 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 72.9% | 0.349 | BTCUSDT | 46.5K | 106.6% |
| 219 | ESP | ESPUSDT (spot) | spot, usdm-futures | 72.8% | 0.400 | INUSDT | 15.3K | 93.0% |
| 220 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 72.8% | 0.344 | BTCUSDT | 120.3K | 147.1% |
| 221 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 72.7% | 0.395 | BTCUSDT | 30.3K | 98.8% |
| 222 | BZ | BZUSDT (usdm-futures) | usdm-futures | 72.6% | 0.406 | XLEUSDT | 4.2M | 44.6% |
| 223 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 72.4% | 0.402 | BTCUSDT | 61.8K | 75.2% |
| 224 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 72.2% | 0.469 | BTCUSDT | 15.7K | 70.1% |
| 225 | WET | WETUSDT (usdm-futures) | usdm-futures | 72.1% | 0.361 | BTCUSDT | 52.9K | 107.9% |
| 226 | UBER | UBERUSDT (usdm-futures) | usdm-futures | 72.0% | 0.359 | DKNGUSDT | 4.4K | 39.5% |
| 227 | BOB | 1000000BOBUSDT (usdm-futures) | usdm-futures | 71.8% | 0.380 | BTCUSDT | 22.4K | 88.1% |
| 228 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 71.4% | 0.447 | BTCUSDT | 119.9K | 54.1% |
| 229 | BBX | BBXUSDT (usdm-futures) | usdm-futures | 71.3% | 0.248 | BXUSDT | 61.7K | 116.9% |
| 230 | ADX | ADXUSDT (spot) | spot | 71.2% | 0.425 | BTCUSDT | 6.8K | 63.1% |
| 231 | VANA | VANAUSDT (spot) | spot, usdm-futures | 71.1% | 0.434 | BTCUSDT | 15.2K | 80.3% |
| 232 | APE | APEUSDT (spot) | spot, usdm-futures | 71.0% | 0.401 | BTCUSDT | 37.4K | 94.2% |
| 233 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 70.8% | 0.445 | BTCUSDT | 34.3K | 98.0% |
| 234 | KAT | KATUSDT (spot) | spot, usdm-futures | 70.6% | 0.383 | BTCUSDT | 24.4K | 102.7% |
| 235 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 70.5% | 0.234 | BTCUSDT | 13.3K | 207.3% |
| 236 | CHILLGUY | CHILLGUYUSDT (usdm-futures) | usdm-futures | 70.3% | 0.454 | BTCUSDT | 31.2K | 115.6% |
| 237 | MINA | MINAUSDT (spot) | spot, usdm-futures | 70.2% | 0.478 | BTCUSDT | 11.6K | 92.5% |
| 238 | HPE | HPEUSDT (usdm-futures) | usdm-futures | 70.0% | 0.326 | SPCXBUSDT | 12.5K | 70.4% |
| 239 | YB | YBUSDT (spot) | spot, usdm-futures | 70.0% | 0.395 | BTCUSDT | 7.9K | 109.9% |
| 240 | LA | LAUSDT (spot) | spot, usdm-futures | 69.8% | 0.403 | ERAUSDT | 20.5K | 123.0% |
| 241 | SC | SCUSDT (spot) | spot | 69.5% | 0.456 | BTCUSDT | 3.7K | 63.7% |
| 242 | FLOCK | FLOCKUSDT (usdm-futures) | usdm-futures | 69.3% | 0.459 | BTCUSDT | 38.5K | 117.2% |
| 243 | RED | REDUSDT (spot) | spot, usdm-futures | 69.2% | 0.448 | BTCUSDT | 14K | 99.8% |
| 244 | BICO | BICOUSDT (spot) | spot, usdm-futures | 69.1% | 0.328 | BTCUSDT | 36.4K | 151.3% |
| 245 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 68.9% | 0.354 | BTCUSDT | 35.7K | 122.7% |
| 246 | A | AUSDT (spot) | spot, usdm-futures | 68.7% | 0.472 | BTCUSDT | 9.9K | 82.9% |
| 247 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 68.5% | 0.391 | BTCUSDT | 89.9K | 117.0% |
| 248 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 68.4% | 0.347 | BTCUSDT | 77.6K | 132.5% |
| 249 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 68.3% | 0.450 | BTCUSDT | 2.9K | 67.8% |
| 250 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 68.3% | 0.305 | COPPERUSDT | 26.9K | 80.0% |
| 251 | HOLO | HOLOUSDT (spot) | spot, usdm-futures | 68.1% | 0.374 | BTCUSDT | 24.3K | 93.9% |
| 252 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 68.0% | 0.266 | CRWDUSDT | 312.2K | 33.9% |
| 253 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 67.8% | 0.436 | BTCUSDT | 891.7K | 58.6% |
| 254 | MEME | MEMEUSDT (spot) | spot, usdm-futures | 67.7% | 0.419 | BTCUSDT | 15.3K | 84.8% |
| 255 | BREV | BREVUSDT (spot) | spot, usdm-futures | 67.5% | 0.330 | WETUSDT | 8.8K | 139.5% |
| 256 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 67.4% | 0.450 | BTCUSDT | 50.9K | 114.1% |
| 257 | XTZ | XTZUSDT (spot) | spot, usdm-futures | 67.2% | 0.539 | BTCUSDT | 11.6K | 78.7% |
| 258 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 67.0% | 0.461 | BTCUSDT | 8.5K | 86.3% |
| 259 | KAS | KASUSDT (usdm-futures) | usdm-futures | 66.9% | 0.512 | BTCUSDT | 76.1K | 63.8% |
| 260 | FLNC | FLNCUSDT (usdm-futures) | usdm-futures | 66.7% | 0.356 | BTCUSDT | 47.4K | 113.2% |

## Diagnostics

- Basis size selected: 260
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.082
- Maximum pairwise absolute correlation: 0.539
- Mean whole-market projection R²: 82.9%
- Median whole-market projection R²: 80.2%
- 10th-percentile whole-market projection R²: 61.0%
- Minimum whole-market projection R²: 55.8%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.3% | 17.5% | 1.3% | 0.0% |
| 5 | 23.7% | 18.5% | 2.4% | 0.1% |
| 10 | 25.4% | 20.8% | 3.5% | 0.8% |
| 15 | 26.9% | 22.0% | 4.6% | 1.5% |
| 20 | 28.4% | 23.2% | 5.5% | 2.2% |
| 25 | 30.1% | 24.6% | 6.6% | 3.2% |
| 30 | 31.6% | 26.0% | 7.7% | 4.0% |
| 35 | 33.2% | 27.3% | 8.7% | 5.5% |
| 40 | 34.6% | 28.9% | 10.0% | 6.3% |
| 45 | 36.2% | 30.7% | 11.1% | 7.6% |
| 50 | 37.5% | 31.7% | 12.1% | 8.8% |
| 55 | 38.9% | 32.8% | 13.2% | 9.8% |
| 60 | 40.3% | 34.3% | 13.8% | 10.7% |
| 65 | 42.0% | 35.8% | 15.4% | 11.8% |
| 70 | 43.7% | 37.3% | 16.4% | 13.1% |
| 75 | 45.4% | 39.1% | 17.9% | 14.2% |
| 80 | 46.9% | 40.2% | 19.0% | 15.1% |
| 85 | 48.3% | 41.5% | 20.2% | 16.1% |
| 90 | 49.5% | 42.5% | 21.6% | 17.0% |
| 95 | 50.9% | 44.1% | 22.7% | 18.1% |
| 100 | 52.2% | 45.3% | 23.5% | 18.8% |
| 105 | 53.4% | 46.3% | 25.4% | 19.8% |
| 110 | 54.7% | 47.2% | 26.3% | 20.9% |
| 115 | 55.8% | 48.9% | 27.5% | 22.1% |
| 120 | 57.0% | 50.1% | 28.6% | 23.0% |
| 125 | 58.1% | 51.3% | 30.2% | 24.2% |
| 130 | 59.4% | 53.0% | 31.6% | 25.8% |
| 135 | 60.5% | 54.1% | 32.5% | 26.9% |
| 140 | 61.5% | 54.7% | 33.8% | 28.4% |
| 145 | 62.6% | 56.0% | 35.0% | 29.6% |
| 150 | 63.8% | 57.4% | 36.5% | 30.9% |
| 155 | 64.8% | 58.2% | 37.7% | 31.7% |
| 160 | 65.8% | 59.0% | 39.0% | 32.6% |
| 165 | 66.8% | 60.0% | 40.0% | 34.3% |
| 170 | 67.9% | 61.4% | 41.1% | 35.7% |
| 175 | 69.0% | 62.3% | 42.4% | 36.6% |
| 180 | 70.0% | 63.4% | 43.2% | 37.8% |
| 185 | 71.1% | 64.8% | 44.5% | 39.3% |
| 190 | 72.0% | 66.0% | 45.3% | 40.9% |
| 195 | 72.9% | 66.9% | 47.2% | 42.0% |
| 200 | 73.7% | 68.0% | 48.1% | 43.7% |
| 205 | 74.6% | 68.7% | 49.4% | 44.7% |
| 210 | 75.4% | 69.8% | 50.5% | 45.7% |
| 215 | 76.2% | 70.9% | 51.6% | 46.2% |
| 220 | 76.9% | 72.0% | 52.7% | 47.1% |
| 225 | 77.7% | 73.1% | 53.7% | 48.2% |
| 230 | 78.6% | 74.1% | 54.9% | 49.4% |
| 235 | 79.3% | 75.1% | 55.9% | 50.6% |
| 240 | 80.2% | 76.3% | 57.0% | 51.7% |
| 245 | 80.8% | 77.2% | 57.8% | 52.7% |
| 250 | 81.5% | 77.9% | 58.7% | 53.7% |
| 255 | 82.2% | 78.8% | 59.4% | 54.6% |
| 260 | 82.9% | 80.2% | 61.0% | 55.8% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | ATM | TAIKO | HD | DEXE | US | MAGMA | HEI | XNO | ADBE | EVAA | PIVX | ALCH | BAS | AKE | ZEREBRO | DODO | HANA | M | B | MANTA | EPIC | XPIN | NATGAS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.005 | 0.015 | -0.015 | -0.016 | 0.020 | 0.025 | 0.006 | 0.040 | 0.035 | 0.046 | 0.056 | 0.090 | 0.058 | 0.064 | 0.052 | 0.106 | 0.080 | 0.074 | 0.022 | 0.116 | 0.041 | 0.113 | -0.033 |
| ATM | 0.005 | 1.000 | -0.005 | 0.006 | -0.010 | -0.012 | -0.021 | -0.036 | -0.004 | -0.030 | 0.030 | 0.030 | 0.008 | 0.066 | 0.011 | 0.023 | -0.026 | 0.001 | 0.021 | -0.027 | -0.045 | -0.058 | -0.001 | 0.026 |
| TAIKO | 0.015 | -0.005 | 1.000 | -0.006 | 0.000 | -0.003 | 0.022 | -0.025 | -0.018 | -0.042 | 0.022 | -0.015 | -0.005 | 0.026 | 0.019 | -0.004 | 0.014 | -0.011 | 0.042 | -0.011 | -0.001 | 0.036 | 0.062 | -0.001 |
| HD | -0.015 | 0.006 | -0.006 | 1.000 | -0.003 | -0.010 | -0.016 | -0.027 | -0.030 | 0.023 | 0.008 | 0.011 | 0.007 | 0.040 | 0.007 | 0.021 | -0.022 | 0.021 | 0.047 | 0.047 | -0.016 | 0.013 | -0.016 | -0.027 |
| DEXE | -0.016 | -0.010 | 0.000 | -0.003 | 1.000 | 0.025 | 0.010 | 0.023 | 0.045 | 0.003 | 0.017 | 0.021 | -0.010 | 0.005 | -0.020 | -0.002 | -0.011 | -0.006 | -0.013 | 0.048 | -0.009 | 0.020 | 0.045 | 0.004 |
| US | 0.020 | -0.012 | -0.003 | -0.010 | 0.025 | 1.000 | 0.028 | -0.014 | 0.022 | -0.008 | 0.029 | 0.000 | 0.037 | 0.013 | 0.060 | -0.044 | 0.003 | 0.054 | 0.041 | 0.031 | 0.002 | 0.056 | 0.001 | -0.069 |
| MAGMA | 0.025 | -0.021 | 0.022 | -0.016 | 0.010 | 0.028 | 1.000 | 0.018 | -0.023 | 0.041 | -0.032 | -0.053 | -0.000 | 0.001 | -0.011 | -0.011 | 0.019 | 0.020 | -0.024 | 0.001 | 0.000 | 0.015 | -0.009 | 0.042 |
| HEI | 0.006 | -0.036 | -0.025 | -0.027 | 0.023 | -0.014 | 0.018 | 1.000 | 0.002 | -0.019 | 0.028 | 0.043 | 0.009 | 0.017 | -0.017 | -0.042 | 0.008 | -0.022 | 0.003 | -0.019 | 0.012 | 0.022 | 0.012 | 0.015 |
| XNO | 0.040 | -0.004 | -0.018 | -0.030 | 0.045 | 0.022 | -0.023 | 0.002 | 1.000 | 0.024 | 0.030 | 0.005 | 0.004 | 0.028 | -0.012 | 0.036 | 0.020 | 0.048 | 0.014 | 0.028 | 0.009 | -0.018 | 0.017 | 0.000 |
| ADBE | 0.035 | -0.030 | -0.042 | 0.023 | 0.003 | -0.008 | 0.041 | -0.019 | 0.024 | 1.000 | 0.014 | 0.005 | -0.009 | 0.015 | 0.030 | -0.008 | -0.003 | 0.041 | -0.007 | 0.002 | -0.013 | 0.027 | -0.006 | -0.080 |
| EVAA | 0.046 | 0.030 | 0.022 | 0.008 | 0.017 | 0.029 | -0.032 | 0.028 | 0.030 | 0.014 | 1.000 | 0.007 | 0.012 | 0.009 | 0.012 | 0.051 | 0.002 | 0.019 | -0.005 | 0.095 | 0.033 | -0.021 | 0.024 | -0.020 |
| PIVX | 0.056 | 0.030 | -0.015 | 0.011 | 0.021 | 0.000 | -0.053 | 0.043 | 0.005 | 0.005 | 0.007 | 1.000 | 0.006 | 0.006 | 0.012 | -0.046 | 0.018 | -0.004 | 0.060 | 0.005 | 0.028 | -0.033 | 0.004 | 0.039 |
| ALCH | 0.090 | 0.008 | -0.005 | 0.007 | -0.010 | 0.037 | -0.000 | 0.009 | 0.004 | -0.009 | 0.012 | 0.006 | 1.000 | -0.000 | -0.054 | 0.036 | 0.045 | 0.043 | 0.029 | 0.003 | 0.023 | 0.010 | 0.019 | 0.025 |
| BAS | 0.058 | 0.066 | 0.026 | 0.040 | 0.005 | 0.013 | 0.001 | 0.017 | 0.028 | 0.015 | 0.009 | 0.006 | -0.000 | 1.000 | 0.001 | -0.011 | -0.017 | 0.020 | -0.020 | -0.037 | -0.031 | 0.051 | -0.001 | 0.057 |
| AKE | 0.064 | 0.011 | 0.019 | 0.007 | -0.020 | 0.060 | -0.011 | -0.017 | -0.012 | 0.030 | 0.012 | 0.012 | -0.054 | 0.001 | 1.000 | 0.002 | -0.009 | 0.057 | 0.010 | 0.027 | 0.036 | 0.022 | -0.025 | -0.010 |
| ZEREBRO | 0.052 | 0.023 | -0.004 | 0.021 | -0.002 | -0.044 | -0.011 | -0.042 | 0.036 | -0.008 | 0.051 | -0.046 | 0.036 | -0.011 | 0.002 | 1.000 | 0.009 | 0.025 | 0.004 | 0.036 | 0.015 | -0.039 | 0.013 | 0.025 |
| DODO | 0.106 | -0.026 | 0.014 | -0.022 | -0.011 | 0.003 | 0.019 | 0.008 | 0.020 | -0.003 | 0.002 | 0.018 | 0.045 | -0.017 | -0.009 | 0.009 | 1.000 | 0.003 | 0.053 | 0.001 | 0.036 | 0.053 | 0.062 | 0.026 |
| HANA | 0.080 | 0.001 | -0.011 | 0.021 | -0.006 | 0.054 | 0.020 | -0.022 | 0.048 | 0.041 | 0.019 | -0.004 | 0.043 | 0.020 | 0.057 | 0.025 | 0.003 | 1.000 | 0.002 | 0.011 | 0.041 | -0.029 | -0.013 | -0.015 |
| M | 0.074 | 0.021 | 0.042 | 0.047 | -0.013 | 0.041 | -0.024 | 0.003 | 0.014 | -0.007 | -0.005 | 0.060 | 0.029 | -0.020 | 0.010 | 0.004 | 0.053 | 0.002 | 1.000 | 0.013 | -0.006 | -0.006 | 0.021 | -0.045 |
| B | 0.022 | -0.027 | -0.011 | 0.047 | 0.048 | 0.031 | 0.001 | -0.019 | 0.028 | 0.002 | 0.095 | 0.005 | 0.003 | -0.037 | 0.027 | 0.036 | 0.001 | 0.011 | 0.013 | 1.000 | 0.033 | -0.048 | 0.022 | -0.048 |
| MANTA | 0.116 | -0.045 | -0.001 | -0.016 | -0.009 | 0.002 | 0.000 | 0.012 | 0.009 | -0.013 | 0.033 | 0.028 | 0.023 | -0.031 | 0.036 | 0.015 | 0.036 | 0.041 | -0.006 | 0.033 | 1.000 | 0.022 | 0.048 | 0.012 |
| EPIC | 0.041 | -0.058 | 0.036 | 0.013 | 0.020 | 0.056 | 0.015 | 0.022 | -0.018 | 0.027 | -0.021 | -0.033 | 0.010 | 0.051 | 0.022 | -0.039 | 0.053 | -0.029 | -0.006 | -0.048 | 0.022 | 1.000 | 0.021 | 0.017 |
| XPIN | 0.113 | -0.001 | 0.062 | -0.016 | 0.045 | 0.001 | -0.009 | 0.012 | 0.017 | -0.006 | 0.024 | 0.004 | 0.019 | -0.001 | -0.025 | 0.013 | 0.062 | -0.013 | 0.021 | 0.022 | 0.048 | 0.021 | 1.000 | -0.003 |
| NATGAS | -0.033 | 0.026 | -0.001 | -0.027 | 0.004 | -0.069 | 0.042 | 0.015 | 0.000 | -0.080 | -0.020 | 0.039 | 0.025 | 0.057 | -0.010 | 0.025 | 0.026 | -0.015 | -0.045 | -0.048 | 0.012 | 0.017 | -0.003 | 1.000 |

The complete 260 × 260 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| INX | INXUSDT | 55.8% | 66.5% | BTCUSDT | 0.419 |
| USELESS | USELESSUSDT | 56.0% | 66.3% | BTCUSDT | 0.455 |
| VIC | VICUSDT | 56.2% | 66.2% | BTCUSDT | 0.400 |
| RIVN | RIVNUSDT | 56.3% | 66.1% | CRWDUSDT | -0.304 |
| ALICE | ALICEUSDT | 56.4% | 66.0% | ACEUSDT | 0.437 |
| AMZN | AMZNUSDT | 56.5% | 65.9% | AAPLUSDT | 0.343 |
| USTC | USTCUSDT | 56.6% | 65.9% | BTCUSDT | 0.462 |
| VELODROME | VELODROMEUSDT | 56.9% | 65.7% | BTCUSDT | 0.507 |
| PUMP | PUMPUSDT | 56.9% | 65.6% | BTCUSDT | 0.525 |
| MAGIC | MAGICUSDT | 57.0% | 65.6% | BTCUSDT | 0.466 |
| 2Z | 2ZUSDT | 57.0% | 65.6% | BTCUSDT | 0.483 |
| TRB | TRBUSDT | 57.0% | 65.6% | XTZUSDT | 0.474 |
| PYTH | PYTHUSDT | 57.0% | 65.5% | BTCUSDT | 0.455 |
| FLUID | FLUIDUSDT | 57.1% | 65.5% | BTCUSDT | 0.493 |
| AIA | AIAUSDT | 57.4% | 65.2% | BTCUSDT | 0.496 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

