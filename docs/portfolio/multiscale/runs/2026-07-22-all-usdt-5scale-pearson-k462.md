# Binance multiscale portfolio basis

Generated 2026-07-23T18:39:51.754Z.

## Scope and method

- Scale views: 365d × 1d, 30d × 4h, 14d × 1h, 7d × 15m, 1d × 1m
- Union universe: 700 economic assets
- Selected basis: 462 assets
- Compression: 34.0%
- Coverage target reached: yes
- Construction: equal-weight direct sum of the five standardized correlation views, followed by column-pivoted QR
- Missing histories contribute no scale block, so persistent multi-horizon assets receive more selection norm than one-scale-only assets

## Coverage by scale

| Scale | Samples | Eligible | Individual basis | Joint basis available | Median R² | P10 R² | Unselected median | Unselected P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 365d × 1d | 365 | 406 | 117 | 283 | 100.0% | 93.9% | 95.2% | 92.0% |
| 30d × 4h | 180 | 638 | 115 | 454 | 100.0% | 100.0% | 100.0% | 100.0% |
| 14d × 1h | 336 | 671 | 210 | 462 | 100.0% | 100.0% | 100.0% | 100.0% |
| 7d × 15m | 672 | 691 | 321 | 462 | 100.0% | 81.8% | 84.2% | 77.9% |
| 1d × 1m | 1440 | 700 | 506 | 462 | 100.0% | 41.1% | 45.3% | 37.1% |

## Joint diagnostics

- Joint median R²: 100.0%
- Joint lower-decile R²: 50.1%
- Unselected joint median R²: 58.0%
- Unselected joint lower-decile R²: 26.0%
- Mean pairwise absolute joint correlation: 0.102
- Maximum pairwise absolute joint correlation: 0.562

## Selected assets

| # | Asset | Residual | Available scales | Selected by individual scales |
| -: | --- | ---: | --- | --- |
| 1 | BTC | 100.0% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 2 | VELVET | 100.0% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 3 | SYN | 99.9% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 4 | BULLA | 99.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 5 | ATM | 99.6% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 6 | BR | 99.6% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 7 | JST | 99.5% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 8 | M | 99.3% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 9 | HOME | 99.3% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 10 | TAC | 99.1% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 11 | XNO | 98.8% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 12 | AIN | 98.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 13 | RIF | 98.6% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 14 | B2 | 98.6% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 15 | EPIC | 98.2% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 16 | AGT | 98.1% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 17 | B | 98.0% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 18 | PROM | 97.9% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 19 | PIVX | 97.8% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 20 | BAN | 97.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 21 | AIOT | 97.4% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 22 | RATS | 97.3% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 23 | IDOL | 97.1% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 24 | ICNT | 96.9% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 25 | GPS | 96.8% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 26 | DODO | 96.1% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 27 | ALCH | 95.9% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 28 | H | 95.9% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 29 | PARTI | 95.6% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 30 | ZEREBRO | 95.4% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 31 | PORTO | 95.3% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 32 | HEI | 95.1% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 33 | TA | 94.6% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 34 | PYR | 94.1% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 35 | SKYAI | 93.8% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 36 | SUN | 93.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 37 | DCR | 93.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 38 | JELLYJELLY | 93.2% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 39 | TLM | 92.9% | structural, swing, adaptive, intraday, microstructure | structural, swing, intraday, microstructure |
| 40 | RESOLV | 92.8% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 41 | AERGO | 92.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 42 | BEL | 92.6% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 43 | KAITO | 92.5% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 44 | DGB | 92.2% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 45 | TNSR | 92.1% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 46 | CROSS | 91.9% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 47 | AWE | 91.7% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 48 | SKL | 91.7% | structural, swing, adaptive, intraday, microstructure | structural, swing, intraday, microstructure |
| 49 | AVAAI | 91.6% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 50 | BROCCOLIF3B | 91.5% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 51 | SIREN | 91.0% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 52 | ARC | 90.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 53 | XEC | 90.3% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 54 | SXT | 90.2% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 55 | STRAX | 90.1% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 56 | HUMA | 89.7% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 57 | TUT | 89.5% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 58 | MYX | 89.1% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 59 | QKC | 88.8% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 60 | DEXE | 88.4% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 61 | TAIKO | 88.2% | structural, swing, adaptive, intraday, microstructure | structural, swing, intraday, microstructure |
| 62 | LUMIA | 87.9% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 63 | QUICK | 87.8% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 64 | PAXG | 87.5% | structural, swing, adaptive, intraday, microstructure | structural |
| 65 | THE | 87.4% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 66 | ACE | 87.1% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 67 | SOON | 87.0% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 68 | G | 86.9% | structural, swing, adaptive, intraday, microstructure | structural, swing, intraday, microstructure |
| 69 | CATI | 86.5% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 70 | WIN | 86.2% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 71 | HMSTR | 85.8% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 72 | AI | 85.6% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 73 | OGN | 85.4% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday |
| 74 | FHE | 85.2% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 75 | BAR | 85.1% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 76 | LA | 84.8% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 77 | SPELL | 84.5% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 78 | REQ | 84.3% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 79 | MBL | 84.3% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 80 | SQD | 84.1% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 81 | NMR | 83.8% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 82 | TST | 83.4% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 83 | RPL | 83.3% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 84 | AGLD | 83.0% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 85 | MAVIA | 82.8% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive, intraday, microstructure |
| 86 | FTT | 82.6% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 87 | SOLV | 82.4% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 88 | T | 82.2% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 89 | MANTA | 81.6% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 90 | ALPINE | 81.5% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 91 | ACT | 81.4% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 92 | STG | 81.1% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 93 | PROMPT | 80.6% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, microstructure |
| 94 | ARPA | 80.5% | structural, swing, adaptive, intraday, microstructure | structural, swing, intraday, microstructure |
| 95 | PORTAL | 80.4% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 96 | IQ | 80.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 97 | STO | 80.2% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 98 | ID | 79.7% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 99 | BANANAS31 | 79.5% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 100 | SPK | 79.3% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 101 | ONE | 79.2% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 102 | XMR | 79.0% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 103 | GNS | 78.9% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 104 | KMNO | 78.8% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 105 | KOMA | 78.7% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 106 | C | 78.5% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 107 | JTO | 78.3% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 108 | VANA | 77.9% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 109 | VANRY | 77.9% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, microstructure |
| 110 | ONG | 77.5% | structural, swing, adaptive, intraday, microstructure | swing, intraday |
| 111 | AUDIO | 77.4% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 112 | SAFE | 77.3% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 113 | QI | 77.1% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 114 | OG | 76.9% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 115 | BLUR | 76.8% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 116 | HOT | 76.7% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 117 | VIC | 76.2% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 118 | TRX | 76.0% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 119 | DOOD | 75.9% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 120 | BICO | 75.8% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 121 | ACX | 75.7% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 122 | GUN | 75.4% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 123 | COOKIE | 75.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 124 | KGST | 93.9% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 125 | MUBARAK | 74.9% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 126 | OSMO | 74.4% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 127 | SAHARA | 74.3% | structural, swing, adaptive, intraday, microstructure | structural, swing, microstructure |
| 128 | KAVA | 74.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 129 | PIPPIN | 73.8% | structural, swing, adaptive, intraday, microstructure | structural, swing, adaptive |
| 130 | SNX | 73.7% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 131 | ERA | 73.4% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 132 | NIL | 73.4% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 133 | GWEI | 91.4% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 134 | USTC | 73.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 135 | BTW | 91.3% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 136 | XNY | 91.0% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 137 | V | 90.9% | swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 138 | APE | 72.7% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 139 | XVG | 72.6% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 140 | PSG | 72.3% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 141 | ADX | 72.2% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 142 | GNO | 72.1% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 143 | JASMY | 71.9% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 144 | ZBT | 89.9% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 145 | AMP | 71.8% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 146 | U | 89.7% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 147 | CRWD | 89.6% | swing, adaptive, intraday, microstructure | swing, intraday |
| 148 | 龙虾 | 89.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 149 | BOB | 71.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 150 | ZRO | 71.4% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 151 | SCRT | 71.3% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 152 | KGEN | 89.0% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 153 | XPIN | 88.8% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 154 | US | 88.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 155 | CYS | 88.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 156 | XAN | 88.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 157 | ZEST | 88.3% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 158 | BLUAI | 88.2% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 159 | KERNEL | 70.4% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 160 | HFT | 70.2% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 161 | PHAROS | 87.8% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 162 | NATGAS | 87.7% | swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 163 | MAGMA | 87.7% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 164 | FF | 87.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 165 | ASR | 69.9% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 166 | BEAT | 87.3% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 167 | ALLO | 87.1% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 168 | BAS | 87.1% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 169 | TAG | 87.1% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 170 | MMT | 87.0% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 171 | NFLX | 86.7% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 172 | BABY | 69.3% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 173 | CTSI | 69.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 174 | VVV | 69.0% | structural, swing, adaptive, intraday, microstructure | structural |
| 175 | PHA | 69.0% | structural, swing, adaptive, intraday, microstructure | structural, adaptive, intraday, microstructure |
| 176 | SLX | 86.2% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 177 | GLMR | 86.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 178 | CLO | 85.9% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 179 | 币安人生 | 85.7% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 180 | PUMPBTC | 68.6% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 181 | CITY | 68.5% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 182 | PYTH | 68.4% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 183 | FOLKS | 85.4% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 184 | XVS | 68.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 185 | BTTC | 85.3% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 186 | RED | 68.0% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 187 | JCT | 84.9% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 188 | ON | 84.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 189 | MINA | 67.8% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 190 | LYN | 84.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 191 | TWT | 67.6% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 192 | EVAA | 84.5% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 193 | EGLD | 67.5% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, microstructure |
| 194 | SWARMS | 67.4% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 195 | QNT | 67.4% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 196 | TRUTH | 84.2% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 197 | GRASS | 67.3% | structural, swing, adaptive, intraday, microstructure | - |
| 198 | HYPER | 67.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 199 | SENT | 83.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 200 | BRKB | 83.6% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 201 | TRIA | 83.4% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 202 | Q | 83.1% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 203 | GUA | 83.0% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 204 | MELANIA | 66.2% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 205 | NVO | 82.6% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 206 | ZAMA | 82.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 207 | ACU | 82.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 208 | BIO | 65.9% | structural, swing, adaptive, intraday, microstructure | structural, intraday, microstructure |
| 209 | UAI | 82.3% | swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 210 | LISTA | 65.6% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 211 | STABLE | 82.0% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 212 | FORM | 65.5% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 213 | CTR | 81.8% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 214 | BERA | 65.4% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 215 | ESP | 81.6% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 216 | EBAY | 81.4% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 217 | ANKR | 65.0% | structural, swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 218 | NIGHT | 81.0% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 219 | APR | 80.9% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 220 | ENSO | 80.7% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 221 | CC | 80.7% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 222 | KITE | 80.5% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 223 | TAKE | 80.4% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 224 | XLE | 80.3% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 225 | SC | 64.1% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 226 | AIO | 80.1% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 227 | PAYP | 80.0% | swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 228 | AKE | 80.0% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 229 | ATH | 63.9% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 230 | YFI | 63.8% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 231 | KAIA | 63.7% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 232 | SONY | 79.6% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 233 | OPENAI | 79.5% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 234 | TRADOOR | 79.3% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 235 | BANANA | 63.4% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 236 | NAORIS | 79.2% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 237 | TFUEL | 63.4% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 238 | AUCTION | 78.9% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 239 | NEXO | 63.1% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 240 | UB | 78.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 241 | EWZ | 78.6% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 242 | ROBO | 78.4% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 243 | NOM | 78.3% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 244 | RECALL | 78.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 245 | YB | 78.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 246 | 4 | 78.0% | swing, adaptive, intraday, microstructure | swing, adaptive, microstructure |
| 247 | POWER | 78.0% | swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 248 | LAB | 77.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 249 | AT | 77.5% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 250 | BIRB | 77.4% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 251 | MORPHO | 77.1% | swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 252 | AIGENSYN | 77.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 253 | OPN | 77.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 254 | SKR | 77.0% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 255 | WET | 76.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 256 | MITO | 76.8% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 257 | KNC | 61.4% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 258 | RARE | 76.7% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 259 | COLLECT | 76.6% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 260 | TOWNS | 76.4% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 261 | PTB | 76.3% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 262 | BASED | 76.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 263 | ELSA | 76.1% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 264 | AAPL | 75.9% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 265 | EDU | 60.7% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 266 | STAR | 75.8% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 267 | JUV | 60.4% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 268 | AIA | 75.4% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 269 | SPCX | 75.4% | swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 270 | RE | 75.4% | swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 271 | CELO | 60.2% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 272 | WLFI | 75.2% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 273 | MOVR | 60.1% | structural, swing, adaptive, intraday, microstructure | structural |
| 274 | MERL | 59.9% | structural, swing, adaptive, intraday, microstructure | structural |
| 275 | BLESS | 74.7% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 276 | SPACE | 74.6% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 277 | SIGN | 59.6% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 278 | PRL | 74.3% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 279 | DRIFT | 59.4% | structural, swing, adaptive, intraday, microstructure | structural |
| 280 | ARIA | 74.1% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 281 | STORJ | 59.3% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 282 | LLY | 73.7% | swing, adaptive, intraday, microstructure | swing, microstructure |
| 283 | TREE | 73.7% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 284 | MAV | 58.9% | structural, swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 285 | MET | 73.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 286 | HD | 73.4% | swing, adaptive, intraday, microstructure | microstructure |
| 287 | GENIUS | 73.2% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 288 | FIGHT | 73.1% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 289 | IOTX | 58.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 290 | MEME | 58.2% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 291 | POL | 58.1% | structural, swing, adaptive, intraday, microstructure | adaptive, intraday |
| 292 | ALICE | 58.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 293 | ESPORTS | 72.5% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 294 | GMX | 57.9% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 295 | GME | 72.1% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 296 | TKO | 57.7% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 297 | FRAX | 72.0% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 298 | HYUNDAI | 71.8% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 299 | JPM | 71.6% | swing, adaptive, intraday, microstructure | microstructure |
| 300 | RAVE | 71.5% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 301 | ENJ | 57.1% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 302 | ETHFI | 57.0% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 303 | SAPIEN | 71.2% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 304 | ZIL | 56.8% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 305 | UBER | 70.9% | swing, adaptive, intraday, microstructure | adaptive |
| 306 | MOVE | 56.7% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 307 | MANTRA | 70.7% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 308 | STEEM | 56.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 309 | EDGE | 70.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 310 | BSB | 70.4% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 311 | 0G | 70.2% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 312 | SANTOS | 56.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 313 | BAND | 56.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 314 | CHILLGUY | 56.0% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 315 | GLM | 56.0% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 316 | PLAY | 69.9% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 317 | SAGA | 55.7% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 318 | LIGHT | 69.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 319 | HAEDAL | 55.5% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 320 | GTC | 69.2% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 321 | BSV | 55.3% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 322 | BILL | 69.0% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 323 | KAT | 69.0% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 324 | INIT | 55.0% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 325 | LDO | 54.9% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 326 | POLYX | 54.9% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 327 | DUSK | 54.9% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 328 | CAT | 54.8% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 329 | FLOCK | 68.3% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 330 | CGPT | 54.6% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 331 | ME | 54.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 332 | SLP | 54.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 333 | BONK | 54.3% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 334 | NXPC | 54.2% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 335 | A | 67.8% | structural, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 336 | HIMS | 67.6% | swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 337 | COAI | 67.5% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 338 | ORCA | 53.8% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 339 | SOPH | 53.8% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 340 | ZKP | 67.1% | swing, adaptive, intraday, microstructure | microstructure |
| 341 | C98 | 53.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 342 | IOTA | 53.3% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 343 | FOGO | 66.6% | swing, adaptive, intraday, microstructure | swing, microstructure |
| 344 | BANK | 66.5% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 345 | HANA | 66.4% | swing, adaptive, intraday, microstructure | swing, adaptive, microstructure |
| 346 | ARK | 53.0% | structural, swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 347 | MEGA | 66.2% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 348 | SHELL | 52.9% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 349 | MASK | 52.6% | structural, swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 350 | RIVN | 65.6% | swing, adaptive, intraday, microstructure | microstructure |
| 351 | RAD | 65.5% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 352 | CHEEMS | 52.4% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 353 | USUAL | 52.3% | structural, swing, adaptive, intraday, microstructure | - |
| 354 | VELODROME | 52.2% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 355 | MOCA | 52.2% | structural, swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 356 | PIEVERSE | 64.8% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 357 | TSLA | 64.7% | swing, adaptive, intraday, microstructure | microstructure |
| 358 | EDEN | 64.6% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 359 | RVN | 51.7% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 360 | MAGIC | 51.6% | structural, swing, adaptive, intraday, microstructure | - |
| 361 | BROCCOLI714 | 51.6% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 362 | DYDX | 51.5% | structural, swing, adaptive, intraday, microstructure | swing, adaptive, microstructure |
| 363 | ONDO | 51.5% | structural, swing, adaptive, intraday, microstructure | - |
| 364 | S | 51.4% | structural, swing, adaptive, intraday, microstructure | - |
| 365 | NEWT | 51.4% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 366 | STBL | 64.1% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 367 | ANTHROPIC | 63.9% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 368 | TURTLE | 63.8% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 369 | IN | 63.7% | swing, adaptive, intraday, microstructure | swing, microstructure |
| 370 | SPORTFUN | 63.5% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 371 | MTL | 50.8% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 372 | FIDA | 50.7% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 373 | PUNDIX | 50.6% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 374 | SKY | 63.2% | swing, adaptive, intraday, microstructure | swing, intraday |
| 375 | ETHW | 50.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 376 | OPG | 62.9% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 377 | CYBER | 50.2% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 378 | TRUMP | 50.1% | structural, swing, adaptive, intraday, microstructure | - |
| 379 | GRIFFAIN | 50.0% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 380 | LIT | 62.4% | swing, adaptive, intraday, microstructure | adaptive |
| 381 | METIS | 49.9% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 382 | OPEN | 62.3% | swing, adaptive, intraday, microstructure | microstructure |
| 383 | AMZN | 62.2% | swing, adaptive, intraday, microstructure | microstructure |
| 384 | FLOW | 49.6% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 385 | LAYER | 49.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 386 | CAP | 82.4% | adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 387 | RLC | 49.4% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 388 | FLNC | 61.7% | swing, adaptive, intraday, microstructure | microstructure |
| 389 | ADBE | 61.5% | swing, adaptive, intraday, microstructure | - |
| 390 | FLUID | 61.4% | swing, adaptive, intraday, microstructure | microstructure |
| 391 | COW | 49.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 392 | ANIME | 48.9% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 393 | AKT | 48.9% | structural, swing, adaptive, intraday, microstructure | - |
| 394 | COMP | 48.7% | structural, swing, adaptive, intraday, microstructure | - |
| 395 | POWR | 48.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 396 | ORDI | 48.5% | structural, swing, adaptive, intraday, microstructure | - |
| 397 | LSK | 48.3% | structural, swing, adaptive, intraday, microstructure | structural, microstructure |
| 398 | API3 | 48.3% | structural, swing, adaptive, intraday, microstructure | structural |
| 399 | HEMI | 60.2% | swing, adaptive, intraday, microstructure | microstructure |
| 400 | BCH | 48.1% | structural, swing, adaptive, intraday, microstructure | - |
| 401 | ILV | 48.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 402 | LUNA | 48.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 403 | TOSHI | 60.0% | swing, adaptive, intraday, microstructure | swing, adaptive, intraday, microstructure |
| 404 | JOE | 48.0% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 405 | INJ | 47.9% | structural, swing, adaptive, intraday, microstructure | - |
| 406 | KSTR | 79.6% | adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 407 | CHR | 47.7% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 408 | PLUME | 59.6% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 409 | QTUM | 47.6% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 410 | SMCI | 59.5% | swing, adaptive, intraday, microstructure | swing, intraday |
| 411 | ZEC | 47.4% | structural, swing, adaptive, intraday, microstructure | structural |
| 412 | HOLO | 59.2% | swing, adaptive, intraday, microstructure | microstructure |
| 413 | KAS | 47.3% | structural, swing, adaptive, intraday, microstructure | - |
| 414 | O | 78.5% | adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 415 | UMA | 47.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 416 | 2Z | 58.6% | swing, adaptive, intraday, microstructure | - |
| 417 | BTR | 58.5% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 418 | CHIP | 58.4% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 419 | WAXP | 46.5% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 420 | CSCO | 58.1% | swing, adaptive, intraday, microstructure | microstructure |
| 421 | ACM | 46.4% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 422 | BREV | 57.9% | swing, adaptive, intraday, microstructure | swing, microstructure |
| 423 | ACH | 46.2% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 424 | TRB | 46.2% | structural, swing, adaptive, intraday, microstructure | - |
| 425 | PUMP | 57.6% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 426 | BX | 57.6% | swing, adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 427 | MON | 57.5% | swing, adaptive, intraday, microstructure | microstructure |
| 428 | BSP | 76.6% | adaptive, intraday, microstructure | intraday, microstructure |
| 429 | ZK | 45.9% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 430 | USELESS | 57.2% | swing, adaptive, intraday, microstructure | intraday |
| 431 | DATAIP | 76.2% | adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 432 | CTK | 45.6% | structural, swing, adaptive, intraday, microstructure | - |
| 433 | VTHO | 45.6% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 434 | EWT | 56.9% | swing, adaptive, intraday, microstructure | microstructure |
| 435 | RUNE | 45.3% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 436 | ALT | 45.2% | structural, swing, adaptive, intraday, microstructure | - |
| 437 | BZ | 56.4% | swing, adaptive, intraday, microstructure | swing, intraday, microstructure |
| 438 | CFX | 45.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 439 | AXL | 45.0% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 440 | BAT | 44.9% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 441 | GAS | 44.9% | structural, swing, adaptive, intraday, microstructure | - |
| 442 | ZORA | 56.1% | swing, adaptive, intraday, microstructure | - |
| 443 | SFP | 44.7% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 444 | DIS | 55.9% | swing, adaptive, intraday, microstructure | adaptive, microstructure |
| 445 | RSR | 44.6% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 446 | REZ | 44.4% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 447 | INX | 55.4% | swing, adaptive, intraday, microstructure | microstructure |
| 448 | DOLO | 55.4% | swing, adaptive, intraday, microstructure | microstructure |
| 449 | ASTER | 55.3% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 450 | BARD | 55.2% | swing, adaptive, intraday, microstructure | - |
| 451 | MANA | 44.1% | structural, swing, adaptive, intraday, microstructure | microstructure |
| 452 | CRV | 44.0% | structural, swing, adaptive, intraday, microstructure | - |
| 453 | IWM | 55.0% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 454 | YGG | 43.9% | structural, swing, adaptive, intraday, microstructure | - |
| 455 | GOOGL | 73.1% | adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 456 | WCT | 43.8% | structural, swing, adaptive, intraday, microstructure | - |
| 457 | HBAR | 43.8% | structural, swing, adaptive, intraday, microstructure | - |
| 458 | ONT | 43.6% | structural, swing, adaptive, intraday, microstructure | structural |
| 459 | ENS | 43.6% | structural, swing, adaptive, intraday, microstructure | intraday, microstructure |
| 460 | ARX | 72.5% | adaptive, intraday, microstructure | adaptive, intraday, microstructure |
| 461 | CFG | 54.2% | swing, adaptive, intraday, microstructure | intraday, microstructure |
| 462 | LQTY | 54.0% | swing, adaptive, intraday, microstructure | microstructure |

## Joint coverage curve

| Size | Mean R² | Median R² | P10 R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.0% | 7.6% | 0.4% | 0.0% |
| 25 | 17.7% | 11.3% | 2.6% | 0.2% |
| 50 | 24.2% | 16.6% | 4.4% | 0.5% |
| 75 | 30.6% | 22.5% | 6.3% | 0.8% |
| 100 | 35.9% | 27.7% | 7.9% | 1.3% |
| 125 | 40.6% | 32.8% | 10.0% | 1.7% |
| 150 | 45.0% | 37.0% | 12.4% | 2.2% |
| 175 | 49.2% | 40.6% | 15.7% | 2.7% |
| 200 | 53.1% | 43.7% | 18.2% | 3.3% |
| 225 | 57.0% | 46.7% | 20.9% | 3.8% |
| 250 | 60.6% | 49.3% | 23.7% | 4.0% |
| 275 | 64.3% | 53.1% | 27.9% | 4.4% |
| 300 | 67.6% | 57.9% | 31.0% | 5.0% |
| 325 | 70.5% | 64.0% | 33.8% | 5.3% |
| 350 | 73.3% | 95.2% | 36.4% | 5.9% |
| 375 | 75.9% | 100.0% | 38.8% | 6.5% |
| 400 | 78.5% | 100.0% | 41.8% | 7.0% |
| 425 | 81.0% | 100.0% | 44.9% | 7.7% |
| 450 | 83.4% | 100.0% | 48.7% | 8.7% |
| 462 | 84.4% | 100.0% | 50.1% | 9.0% |

## Least-covered assets

| Asset | Joint R² | Available scales |
| --- | ---: | --- |
| SHAZ | 9.0% | microstructure |
| MINIMAX | 10.6% | microstructure |
| ZHIPU | 11.2% | microstructure |
| SOFI | 12.2% | microstructure |
| PENG | 14.1% | microstructure |
| PANW | 14.5% | microstructure |
| IBM | 15.8% | intraday, microstructure |
| BABA | 16.0% | intraday, microstructure |
| WEN | 16.1% | intraday, microstructure |
| TZA | 16.3% | microstructure |
| AVGO | 18.8% | intraday, microstructure |
| AERO | 20.0% | microstructure |
| XBI | 20.0% | intraday, microstructure |
| FWDI | 20.5% | intraday, microstructure |
| BOT | 21.1% | intraday, microstructure |
| BNC | 21.1% | intraday, microstructure |
| NOK | 21.3% | intraday, microstructure |
| GEV | 22.2% | intraday, microstructure |
| HOOD | 24.1% | intraday, microstructure |
| SNOW | 24.9% | intraday, microstructure |
