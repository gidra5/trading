# Binance multiscale portfolio basis

Generated 2026-07-23T18:42:49.792Z.

## Scope and method

- Scale views: 365d × 1d, 30d × 4h, 14d × 1h, 7d × 15m, 1d × 1m
- Union universe: 700 economic assets
- Selected basis: 177 assets
- Compression: 74.7%
- Sizing: fixed Pareto size
- Construction: equal-weight direct sum of the five standardized correlation views, followed by column-pivoted QR
- Missing histories contribute no scale block, so persistent multi-horizon assets receive more selection norm than one-scale-only assets

## Coverage by scale

| Scale | Samples | Eligible | Individual basis | Joint basis available | Median R² | P10 R² | Unselected median | Unselected P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 365d × 1d | 365 | 406 | 117 | 149 | 86.5% | 72.0% | 80.5% | 68.0% |
| 30d × 4h | 180 | 638 | 115 | 177 | 99.8% | 98.9% | 99.6% | 98.7% |
| 14d × 1h | 336 | 671 | 210 | 177 | 71.0% | 56.9% | 66.3% | 55.8% |
| 7d × 15m | 672 | 691 | 321 | 177 | 51.2% | 31.6% | 45.7% | 29.9% |
| 1d × 1m | 1440 | 700 | 506 | 177 | 20.6% | 13.7% | 18.1% | 13.2% |

## Joint diagnostics

- Joint median R²: 40.7%
- Joint lower-decile R²: 16.0%
- Unselected joint median R²: 32.2%
- Unselected joint lower-decile R²: 14.7%
- Mean pairwise absolute joint correlation: 0.074
- Maximum pairwise absolute joint correlation: 0.442

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
| 177 | 49.5% | 40.7% | 16.0% | 2.8% |

## Least-covered assets

| Asset | Joint R² | Available scales |
| --- | ---: | --- |
| SHAZ | 2.8% | microstructure |
| MINIMAX | 3.4% | microstructure |
| ZHIPU | 3.7% | microstructure |
| SOFI | 3.8% | microstructure |
| PENG | 3.9% | microstructure |
| WEN | 4.3% | intraday, microstructure |
| PANW | 4.9% | microstructure |
| BNC | 5.0% | intraday, microstructure |
| BABA | 5.1% | intraday, microstructure |
| IBM | 5.1% | intraday, microstructure |
| TZA | 5.2% | microstructure |
| FWDI | 5.4% | intraday, microstructure |
| BOT | 6.2% | intraday, microstructure |
| XBI | 6.8% | intraday, microstructure |
| GEV | 7.2% | intraday, microstructure |
| AVGO | 7.3% | intraday, microstructure |
| CAP | 8.0% | adaptive, intraday, microstructure |
| GOOGL | 8.5% | adaptive, intraday, microstructure |
| NOK | 8.6% | intraday, microstructure |
| O | 8.6% | adaptive, intraday, microstructure |
