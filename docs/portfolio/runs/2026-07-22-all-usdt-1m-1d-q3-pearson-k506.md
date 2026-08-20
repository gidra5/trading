# Binance portfolio basis

Generated 2026-07-23T18:25:20.753Z.

## Scope

- Products: all
- Product listings discovered: spot 3659, usdm-futures 846, coinm-futures 30, options 1550
- Listing statuses: TRADING 3677, BREAK 2283, SETTLING 123, PENDING_TRADING 2
- Numeraire: USDT
- Return window: 2026-07-22T00:00Z through 2026-07-22T23:59Z
- Sampling: 1m log returns (1440 samples over 1 days)
- Correlation: pearson
- Sizing: automatic until median R² >= 80.0% and 10th-percentile R² >= 50.0% (maximum 512)
- Full catalog: 6085 listings, 3677 active listings, 1126 deduplicated economic assets
- Return universe: 700 eligible assets from 714 active continuously priced candidates
- Quality filter: complete window, trades or quote volume on at least 50.0% of UTC days, no stale-price run longer than 48 hours
- TradFi session handling: zero-volume off-session candles do not extend stale-price runs
- Stable assets excluded: yes

## Selected basis

The selector uses column-pivoted QR on standardized return vectors. Residual is the fraction of an asset's return-vector norm not explained by earlier basis assets. Correlations are shown as absolute values because both +1 and -1 are linearly redundant, not orthogonal.

| # | Asset | Canonical market | Products | Residual | Max |r| to earlier | Closest earlier | Traded days | Max stale hours | Median daily quote volume | Annualized vol |
| -: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | BTC | BTCUSDT (spot) | spot, usdm-futures, coinm-futures, options | 100.0% | 0.000 | - | 100.0% | 0 | 1.2B | 30.3% |
| 2 | XNY | XNYUSDT (usdm-futures) | usdm-futures | 100.0% | 0.000 | BTCUSDT | 100.0% | 0.1 | 966.7K | 139.9% |
| 3 | CYS | CYSUSDT (usdm-futures) | usdm-futures | 100.0% | 0.004 | XNYUSDT | 100.0% | 0.1 | 907.5K | 90.5% |
| 4 | KGST | KGSTUSDT (spot) | spot | 100.0% | 0.012 | CYSUSDT | 100.0% | 1.4 | 4.6M | 3.3% |
| 5 | T | TUSDT (spot) | spot, usdm-futures | 100.0% | 0.012 | CYSUSDT | 100.0% | 0.5 | 880K | 108.8% |
| 6 | INIT | INITUSDT (spot) | spot, usdm-futures | 100.0% | 0.013 | BTCUSDT | 100.0% | 0.6 | 892.3K | 133.1% |
| 7 | RIF | RIFUSDT (spot) | spot, usdm-futures | 100.0% | 0.016 | BTCUSDT | 100.0% | 0.1 | 8.8M | 724.2% |
| 8 | ANTHROPIC | ANTHROPICUSDT (usdm-futures) | usdm-futures | 100.0% | 0.017 | CYSUSDT | 100.0% | 0 | 15.9M | 149.9% |
| 9 | B | BUSDT (usdm-futures) | usdm-futures | 99.9% | 0.022 | CYSUSDT | 100.0% | 0.1 | 12.1M | 179.5% |
| 10 | IOTX | IOTXUSDT (spot) | spot, usdm-futures | 99.9% | 0.029 | TUSDT | 100.0% | 1.1 | 66.4K | 78.9% |
| 11 | TOWNS | TOWNSUSDT (spot) | spot, usdm-futures | 99.9% | 0.031 | RIFUSDT | 100.0% | 0.4 | 1.5M | 226.2% |
| 12 | BAN | BANUSDT (usdm-futures) | usdm-futures | 99.9% | 0.026 | ANTHROPICUSDT | 100.0% | 0.1 | 2.4M | 79.4% |
| 13 | BEAT | BEATUSDT (usdm-futures) | usdm-futures | 99.9% | 0.031 | INITUSDT | 100.0% | 0.1 | 20.8M | 156.2% |
| 14 | O | OUSDT (usdm-futures) | usdm-futures | 99.9% | 0.032 | BTCUSDT | 100.0% | 0 | 4.8M | 146.1% |
| 15 | BROCCOLIF3B | BROCCOLIF3BUSDT (usdm-futures) | usdm-futures | 99.8% | 0.028 | XNYUSDT | 100.0% | 0.1 | 71.6M | 474.9% |
| 16 | BOT | BOTUSDT (usdm-futures) | usdm-futures | 99.8% | 0.037 | XNYUSDT | 100.0% | 0.1 | 4.1M | 163.0% |
| 17 | QUICK | QUICKUSDT (spot) | spot | 99.8% | 0.037 | XNYUSDT | 100.0% | 0.4 | 28.3K | 155.1% |
| 18 | GPS | GPSUSDT (spot) | spot, usdm-futures | 99.7% | 0.040 | BTCUSDT | 100.0% | 0.5 | 253.4K | 72.4% |
| 19 | PYR | PYRUSDT (spot) | spot | 99.7% | 0.033 | TOWNSUSDT | 100.0% | 1 | 1.9M | 544.6% |
| 20 | NATGAS | NATGASUSDT (usdm-futures) | usdm-futures | 99.7% | 0.037 | BROCCOLIF3BUSDT | 100.0% | 0.1 | 19.7M | 40.8% |
| 21 | HOT | HOTUSDT (spot) | spot, usdm-futures | 99.6% | 0.038 | NATGASUSDT | 100.0% | 1.2 | 56.1K | 67.8% |
| 22 | JPM | JPMUSDT (usdm-futures) | usdm-futures | 99.6% | 0.043 | NATGASUSDT | 100.0% | 0.1 | 588.5K | 22.7% |
| 23 | STAR | STARUSDT (usdm-futures) | usdm-futures | 99.6% | 0.042 | BROCCOLIF3BUSDT | 100.0% | 0 | 1.7M | 172.2% |
| 24 | ATM | ATMUSDT (spot) | spot | 99.5% | 0.047 | IOTXUSDT | 100.0% | 0.2 | 4.8M | 548.8% |
| 25 | V | VUSDT (usdm-futures) | usdm-futures | 99.5% | 0.059 | CYSUSDT | 100.0% | 0.1 | 313.4K | 26.1% |
| 26 | AWE | AWEUSDT (spot) | spot, usdm-futures | 99.5% | 0.049 | TOWNSUSDT | 100.0% | 0.9 | 210.7K | 80.0% |
| 27 | RE | REUSDT (spot) | spot, usdm-futures | 99.5% | 0.049 | XNYUSDT | 100.0% | 0 | 380.1M | 284.1% |
| 28 | NVO | NVOUSDT (usdm-futures) | usdm-futures | 99.4% | 0.056 | BEATUSDT | 100.0% | 0.1 | 333.8K | 46.7% |
| 29 | KSTR | KSTRUSDT (usdm-futures) | usdm-futures | 99.3% | 0.062 | BTCUSDT | 100.0% | 0.2 | 4M | 98.5% |
| 30 | XNO | XNOUSDT (spot) | spot | 99.3% | 0.045 | BUSDT | 100.0% | 0.3 | 1.1M | 523.7% |
| 31 | JUV | JUVUSDT (spot) | spot | 99.2% | 0.055 | INITUSDT | 100.0% | 0.3 | 214.7K | 130.3% |
| 32 | HEI | HEIUSDT (spot) | spot, usdm-futures | 99.2% | 0.060 | TUSDT | 100.0% | 0.1 | 1.9M | 176.1% |
| 33 | WAL | WALUSDT (spot) | spot, usdm-futures | 99.1% | 0.048 | QUICKUSDT | 100.0% | 0.7 | 258.5K | 98.2% |
| 34 | DGB | DGBUSDT (spot) | spot | 99.1% | 0.060 | RIFUSDT | 100.0% | 0.4 | 260K | 235.1% |
| 35 | KITE | KITEUSDT (spot) | spot, usdm-futures | 99.1% | 0.056 | REUSDT | 100.0% | 0.2 | 40.6M | 96.3% |
| 36 | AKE | AKEUSDT (usdm-futures) | usdm-futures | 99.1% | 0.054 | IOTXUSDT | 100.0% | 0 | 185.8M | 361.2% |
| 37 | DODO | DODOUSDT (spot) | spot | 99.0% | 0.048 | HOTUSDT | 100.0% | 0.2 | 7.4M | 442.0% |
| 38 | BTTC | BTTCUSDT (spot) | spot | 98.9% | 0.050 | JPMUSDT | 100.0% | 0.8 | 109.8K | 1099.0% |
| 39 | AIA | AIAUSDT (usdm-futures) | usdm-futures | 98.9% | 0.076 | VUSDT | 100.0% | 0.1 | 14.1M | 248.0% |
| 40 | ZIL | ZILUSDT (spot) | spot, usdm-futures | 98.9% | 0.058 | BUSDT | 100.0% | 0.5 | 1.3M | 159.8% |
| 41 | POLYX | POLYXUSDT (spot) | spot, usdm-futures | 98.9% | 0.082 | BEATUSDT | 100.0% | 1 | 556.7K | 141.3% |
| 42 | H | HUSDT (usdm-futures) | usdm-futures | 98.9% | 0.058 | REUSDT | 100.0% | 0.1 | 2M | 69.3% |
| 43 | BABA | BABABUSDT (spot) | spot, usdm-futures | 98.8% | 0.059 | JPMUSDT | 100.0% | 0 | 508.3K | 72.7% |
| 44 | PARTI | PARTIUSDT (spot) | spot, usdm-futures | 98.7% | 0.057 | VUSDT | 100.0% | 0.6 | 656.7K | 134.1% |
| 45 | SONY | SONYUSDT (usdm-futures) | usdm-futures | 98.7% | 0.061 | JPMUSDT | 100.0% | 0.1 | 155.7K | 31.3% |
| 46 | BLESS | BLESSUSDT (usdm-futures) | usdm-futures | 98.7% | 0.058 | IOTXUSDT | 100.0% | 0 | 31.6M | 457.0% |
| 47 | CAT | 1000CATUSDT (spot) | spot, usdm-futures | 98.6% | 0.059 | BTCUSDT | 100.0% | 0.6 | 80.9K | 241.9% |
| 48 | STRAX | STRAXUSDT (spot) | spot | 98.6% | 0.077 | BTCUSDT | 100.0% | 0.3 | 229K | 75.3% |
| 49 | LISTA | LISTAUSDT (spot) | spot, usdm-futures | 98.6% | 0.071 | BTCUSDT | 100.0% | 0.4 | 4.7M | 273.2% |
| 50 | TAC | TACUSDT (usdm-futures) | usdm-futures | 98.5% | 0.067 | PYRUSDT | 100.0% | 0 | 8.4M | 226.5% |
| 51 | AVGO | AVGOBUSDT (spot) | spot, usdm-futures | 98.5% | 0.073 | INITUSDT | 100.0% | 0.1 | 169.6K | 75.6% |
| 52 | SHAZ | SHAZUSDT (usdm-futures) | usdm-futures | 98.5% | 0.049 | CYSUSDT | 100.0% | 0.1 | 9.3M | 273.2% |
| 53 | CITY | CITYUSDT (spot) | spot | 98.5% | 0.047 | BABABUSDT | 100.0% | 0.3 | 913K | 140.2% |
| 54 | VANRY | VANRYUSDT (spot) | spot, usdm-futures | 98.5% | 0.049 | REUSDT | 100.0% | 0.1 | 2.2M | 172.3% |
| 55 | JST | JSTUSDT (spot) | spot, usdm-futures | 98.4% | 0.058 | SONYUSDT | 100.0% | 0.2 | 2.1M | 47.3% |
| 56 | DEXE | DEXEUSDT (spot) | spot, usdm-futures | 98.3% | 0.081 | ATMUSDT | 100.0% | 0 | 36M | 570.3% |
| 57 | THE | THEUSDT (spot) | spot, usdm-futures | 98.3% | 0.070 | TOWNSUSDT | 100.0% | 0.5 | 1M | 133.9% |
| 58 | XPIN | XPINUSDT (usdm-futures) | usdm-futures | 98.2% | 0.056 | PYRUSDT | 100.0% | 0.1 | 3.6M | 128.3% |
| 59 | LYN | LYNUSDT (usdm-futures) | usdm-futures | 98.2% | 0.073 | BTCUSDT | 100.0% | 0.1 | 1.7M | 111.0% |
| 60 | NOW | NOWUSDT (usdm-futures) | usdm-futures | 98.2% | 0.083 | NVOUSDT | 100.0% | 0.1 | 6.3M | 131.8% |
| 61 | STG | STGUSDT (spot) | spot, usdm-futures | 98.1% | 0.065 | AKEUSDT | 100.0% | 0.4 | 193.1K | 66.2% |
| 62 | PLAY | PLAYUSDT (usdm-futures) | usdm-futures | 98.1% | 0.065 | WALUSDT | 100.0% | 0.1 | 4.4M | 167.7% |
| 63 | EPIC | EPICUSDT (spot) | spot, usdm-futures | 98.0% | 0.075 | TUSDT | 100.0% | 0.1 | 7.2M | 393.9% |
| 64 | COOKIE | COOKIEUSDT (spot) | spot, usdm-futures | 98.0% | 0.076 | CYSUSDT | 100.0% | 1.6 | 76K | 199.7% |
| 65 | PHA | PHAUSDT (spot) | spot, usdm-futures | 97.9% | 0.086 | BTCUSDT | 100.0% | 0.6 | 769.3K | 127.3% |
| 66 | ZEST | ZESTUSDT (usdm-futures) | usdm-futures | 97.9% | 0.071 | IOTXUSDT | 100.0% | 0 | 2M | 136.2% |
| 67 | SYN | SYNUSDT (spot) | spot, usdm-futures | 97.8% | 0.069 | NVOUSDT | 100.0% | 0 | 3.1M | 151.4% |
| 68 | ROBO | ROBOUSDT (spot) | spot, usdm-futures | 97.8% | 0.061 | BTCUSDT | 100.0% | 0.3 | 433.5K | 119.3% |
| 69 | SCRT | SCRTUSDT (spot) | spot, usdm-futures | 97.7% | 0.068 | VUSDT | 100.0% | 0.7 | 237.8K | 100.1% |
| 70 | WET | WETUSDT (usdm-futures) | usdm-futures | 97.7% | 0.105 | BTCUSDT | 100.0% | 0.1 | 807.9K | 67.2% |
| 71 | BTW | BTWUSDT (usdm-futures) | usdm-futures | 97.7% | 0.063 | CYSUSDT | 100.0% | 0 | 9M | 175.1% |
| 72 | NMR | NMRUSDT (spot) | spot, usdm-futures | 97.6% | 0.067 | NATGASUSDT | 100.0% | 0.8 | 98.5K | 46.8% |
| 73 | HOME | HOMEUSDT (spot) | spot, usdm-futures | 97.6% | 0.074 | TUSDT | 100.0% | 0.2 | 4.4M | 176.9% |
| 74 | ILV | ILVUSDT (spot) | spot, usdm-futures | 97.6% | 0.080 | BTCUSDT | 100.0% | 0.8 | 240.4K | 98.5% |
| 75 | TURTLE | TURTLEUSDT (spot) | spot, usdm-futures | 97.6% | 0.063 | BTCUSDT | 100.0% | 0.7 | 683.5K | 131.4% |
| 76 | XEC | XECUSDT (spot) | spot, usdm-futures | 97.5% | 0.074 | BTCUSDT | 100.0% | 0.1 | 3.6M | 224.7% |
| 77 | ANKR | ANKRUSDT (spot) | spot, usdm-futures | 97.5% | 0.080 | BTCUSDT | 100.0% | 1 | 102.9K | 62.0% |
| 78 | SPELL | SPELLUSDT (spot) | spot, usdm-futures | 97.4% | 0.073 | TUSDT | 100.0% | 0.4 | 1.1M | 127.9% |
| 79 | MITO | MITOUSDT (spot) | spot, usdm-futures | 97.4% | 0.081 | BLESSUSDT | 100.0% | 0.1 | 1.3M | 148.7% |
| 80 | TRIA | TRIAUSDT (usdm-futures) | usdm-futures | 97.3% | 0.065 | THEUSDT | 100.0% | 0.1 | 12.3M | 197.5% |
| 81 | VELVET | VELVETUSDT (usdm-futures) | usdm-futures | 97.3% | 0.068 | SYNUSDT | 100.0% | 0.1 | 6.1M | 98.0% |
| 82 | ALICE | ALICEUSDT (spot) | spot, usdm-futures | 97.3% | 0.085 | VUSDT | 100.0% | 0.3 | 2.1M | 132.8% |
| 83 | MAGMA | MAGMAUSDT (usdm-futures) | usdm-futures | 97.3% | 0.072 | BUSDT | 100.0% | 0 | 3.1M | 109.0% |
| 84 | SNX | SNXUSDT (spot) | spot, usdm-futures | 97.2% | 0.083 | BTCUSDT | 100.0% | 0.7 | 2.1M | 173.9% |
| 85 | AIOT | AIOTUSDT (usdm-futures) | usdm-futures | 97.2% | 0.099 | BTCUSDT | 100.0% | 0 | 2.6M | 172.3% |
| 86 | CHEEMS | 1000CHEEMSUSDT (spot) | spot, usdm-futures | 97.1% | 0.060 | BTTCUSDT | 100.0% | 0.1 | 346.6K | 119.2% |
| 87 | TFUEL | TFUELUSDT (spot) | spot | 97.1% | 0.060 | DGBUSDT | 100.0% | 0.4 | 145.6K | 99.4% |
| 88 | WEN | WENUSDT (usdm-futures) | usdm-futures | 97.1% | 0.106 | NVOUSDT | 100.0% | 0.3 | 493.6K | 71.0% |
| 89 | STABLE | STABLEUSDT (usdm-futures) | usdm-futures | 97.0% | 0.087 | KGSTUSDT | 100.0% | 0.1 | 2.1M | 60.2% |
| 90 | LIGHT | LIGHTUSDT (usdm-futures) | usdm-futures | 97.0% | 0.078 | BTCUSDT | 100.0% | 0.2 | 2.1M | 107.6% |
| 91 | C | CUSDT (spot) | spot, usdm-futures | 96.9% | 0.093 | BUSDT | 100.0% | 0.7 | 424.3K | 113.2% |
| 92 | BSP | BSPUSDT (usdm-futures) | usdm-futures | 96.8% | 0.077 | NATGASUSDT | 100.0% | 0.2 | 347.5K | 109.2% |
| 93 | BLUAI | BLUAIUSDT (usdm-futures) | usdm-futures | 96.8% | 0.068 | BROCCOLIF3BUSDT | 100.0% | 0 | 4M | 157.3% |
| 94 | GOOGL | GOOGLBUSDT (spot) | spot, usdm-futures | 96.7% | 0.107 | AVGOBUSDT | 100.0% | 0.1 | 3.3M | 110.1% |
| 95 | CROSS | CROSSUSDT (usdm-futures) | usdm-futures | 96.7% | 0.079 | PHAUSDT | 100.0% | 0.1 | 966.1K | 122.2% |
| 96 | HD | HDUSDT (usdm-futures) | usdm-futures | 96.6% | 0.089 | VUSDT | 100.0% | 0.1 | 268.2K | 42.1% |
| 97 | JCT | JCTUSDT (usdm-futures) | usdm-futures | 96.6% | 0.069 | SCRTUSDT | 100.0% | 0.1 | 2.1M | 159.1% |
| 98 | AERGO | AERGOUSDT (usdm-futures) | usdm-futures | 96.6% | 0.075 | NATGASUSDT | 100.0% | 0.1 | 3.5M | 143.3% |
| 99 | OPENAI | OPENAIUSDT (usdm-futures) | usdm-futures | 96.5% | 0.091 | ANTHROPICUSDT | 100.0% | 0 | 4.5M | 106.0% |
| 100 | RATS | 1000RATSUSDT (usdm-futures) | usdm-futures | 96.5% | 0.077 | EPICUSDT | 100.0% | 0.1 | 2.2M | 111.9% |
| 101 | FF | FFUSDT (spot) | spot, usdm-futures | 96.4% | 0.092 | BTCUSDT | 100.0% | 0.3 | 1.1M | 48.3% |
| 102 | HMSTR | HMSTRUSDT (spot) | spot, usdm-futures | 96.4% | 0.073 | CYSUSDT | 100.0% | 0.1 | 2.2M | 137.4% |
| 103 | ONE | ONEUSDT (spot) | spot, usdm-futures | 96.3% | 0.062 | VANRYUSDT | 100.0% | 0.2 | 3.5M | 450.4% |
| 104 | AIO | AIOUSDT (usdm-futures) | usdm-futures | 96.3% | 0.096 | BTCUSDT | 100.0% | 0 | 1.7M | 124.8% |
| 105 | SOPH | SOPHUSDT (spot) | spot, usdm-futures | 96.2% | 0.084 | AIOTUSDT | 100.0% | 0.4 | 793.1K | 162.6% |
| 106 | COLLECT | COLLECTUSDT (usdm-futures) | usdm-futures | 96.2% | 0.075 | WETUSDT | 100.0% | 0.1 | 1.9M | 152.8% |
| 107 | TAG | TAGUSDT (usdm-futures) | usdm-futures | 96.2% | 0.075 | BTCUSDT | 100.0% | 0.1 | 5.1M | 175.8% |
| 108 | WIN | WINUSDT (spot) | spot | 96.2% | 0.069 | TUSDT | 100.0% | 0.1 | 97.9K | 112.7% |
| 109 | CAP | CAPUSDT (usdm-futures) | usdm-futures | 96.1% | 0.075 | BTCUSDT | 100.0% | 0.1 | 4.8M | 167.3% |
| 110 | SXT | SXTUSDT (spot) | spot, usdm-futures | 96.1% | 0.081 | BSPUSDT | 100.0% | 0.3 | 708K | 130.1% |
| 111 | ICX | ICXUSDT (spot) | spot, usdm-futures | 96.1% | 0.129 | BTCUSDT | 100.0% | 2.6 | 42.4K | 81.9% |
| 112 | EGLD | EGLDUSDT (spot) | spot, usdm-futures | 96.0% | 0.104 | 1000CHEEMSUSDT | 100.0% | 0.7 | 331.9K | 90.2% |
| 113 | WLFI | WLFIUSDT (spot) | spot, usdm-futures | 96.0% | 0.089 | BTCUSDT | 100.0% | 0.3 | 4.3M | 94.5% |
| 114 | STBL | STBLUSDT (usdm-futures) | usdm-futures | 95.9% | 0.094 | BTCUSDT | 100.0% | 0.1 | 2.6M | 130.0% |
| 115 | ZAMA | ZAMAUSDT (spot) | spot, usdm-futures | 95.9% | 0.068 | ZILUSDT | 100.0% | 0.1 | 13.5M | 171.8% |
| 116 | ERA | ERAUSDT (spot) | spot, usdm-futures | 95.8% | 0.090 | REUSDT | 100.0% | 0.1 | 19.9M | 643.9% |
| 117 | OPN | OPNUSDT (spot) | spot, usdm-futures | 95.8% | 0.076 | VANRYUSDT | 100.0% | 0.1 | 15.1M | 172.6% |
| 118 | HIMS | HIMSUSDT (usdm-futures) | usdm-futures | 95.7% | 0.084 | BTCUSDT | 100.0% | 0.1 | 521.5K | 88.0% |
| 119 | DCR | DCRUSDT (spot) | spot | 95.7% | 0.078 | NMRUSDT | 100.0% | 0.4 | 164.5K | 79.3% |
| 120 | ZBT | ZBTUSDT (spot) | spot, usdm-futures | 95.7% | 0.075 | STRAXUSDT | 100.0% | 0.3 | 2M | 126.2% |
| 121 | 币安人生 | 币安人生USDT (spot) | spot, usdm-futures | 95.6% | 0.076 | BTCUSDT | 100.0% | 0.1 | 2.9M | 119.5% |
| 122 | IBM | IBMBUSDT (spot) | spot, usdm-futures | 95.6% | 0.085 | NOWUSDT | 100.0% | 0.2 | 1.3M | 113.5% |
| 123 | NXPC | NXPCUSDT (spot) | spot, usdm-futures | 95.5% | 0.112 | BTCUSDT | 100.0% | 0.3 | 392.9K | 76.6% |
| 124 | TRUTH | TRUTHUSDT (usdm-futures) | usdm-futures | 95.4% | 0.071 | XPINUSDT | 100.0% | 0 | 1.3M | 102.4% |
| 125 | ENSO | ENSOUSDT (spot) | spot, usdm-futures | 95.4% | 0.098 | BTCUSDT | 100.0% | 0.3 | 374.3K | 64.7% |
| 126 | ME | MEUSDT (spot) | spot, usdm-futures | 95.4% | 0.102 | BTCUSDT | 100.0% | 0.5 | 540.5K | 81.9% |
| 127 | NOM | NOMUSDT (spot) | spot, usdm-futures | 95.3% | 0.095 | 1000RATSUSDT | 100.0% | 0.8 | 551.7K | 178.9% |
| 128 | KAITO | KAITOUSDT (spot) | spot, usdm-futures | 95.2% | 0.082 | AIAUSDT | 100.0% | 0.1 | 4M | 142.8% |
| 129 | AIGENSYN | AIGENSYNUSDT (spot) | spot, usdm-futures | 95.2% | 0.106 | BTCUSDT | 100.0% | 0.1 | 796K | 99.8% |
| 130 | ARPA | ARPAUSDT (spot) | spot, usdm-futures | 95.1% | 0.111 | BTCUSDT | 100.0% | 0.8 | 204.2K | 72.3% |
| 131 | BNC | BNCUSDT (usdm-futures) | usdm-futures | 95.1% | 0.074 | BOTUSDT | 100.0% | 0.2 | 745.4K | 99.8% |
| 132 | FOLKS | FOLKSUSDT (usdm-futures) | usdm-futures | 95.0% | 0.102 | LISTAUSDT | 100.0% | 0.1 | 2M | 88.3% |
| 133 | PORTO | PORTOUSDT (spot) | spot | 95.0% | 0.089 | LYNUSDT | 100.0% | 0.2 | 568.2K | 213.9% |
| 134 | JOE | JOEUSDT (spot) | spot, usdm-futures | 95.0% | 0.074 | STGUSDT | 100.0% | 1.2 | 175.3K | 92.6% |
| 135 | BR | BRUSDT (usdm-futures) | usdm-futures | 94.9% | 0.103 | ZBTUSDT | 100.0% | 0 | 2.6M | 171.9% |
| 136 | XAN | XANUSDT (usdm-futures) | usdm-futures | 94.9% | 0.075 | HEIUSDT | 100.0% | 0.1 | 3.6M | 155.5% |
| 137 | BAS | BASUSDT (usdm-futures) | usdm-futures | 94.9% | 0.101 | TACUSDT | 100.0% | 0 | 3.6M | 129.1% |
| 138 | COAI | COAIUSDT (usdm-futures) | usdm-futures | 94.8% | 0.135 | BTCUSDT | 100.0% | 0.1 | 2.5M | 104.3% |
| 139 | PLUME | PLUMEUSDT (spot) | spot, usdm-futures | 94.8% | 0.141 | BTCUSDT | 100.0% | 0.2 | 1.1M | 106.4% |
| 140 | AVAAI | AVAAIUSDT (usdm-futures) | usdm-futures | 94.7% | 0.085 | BTCUSDT | 100.0% | 0 | 14.5M | 278.6% |
| 141 | ACX | ACXUSDT (spot) | spot, usdm-futures | 94.7% | 0.095 | HOTUSDT | 100.0% | 0.8 | 45.5K | 27.1% |
| 142 | AI | AIUSDT (spot) | spot | 94.6% | 0.078 | BUSDT | 100.0% | 0.9 | 208.6K | 153.7% |
| 143 | TKO | TKOUSDT (spot) | spot | 94.6% | 0.077 | WETUSDT | 100.0% | 0.3 | 112.4K | 139.5% |
| 144 | TRX | TRXUSDT (spot) | spot, usdm-futures, coinm-futures | 94.6% | 0.116 | BTCUSDT | 100.0% | 0.2 | 19.8M | 17.2% |
| 145 | EVAA | EVAAUSDT (usdm-futures) | usdm-futures | 94.5% | 0.087 | HOMEUSDT | 100.0% | 0 | 15.1M | 166.9% |
| 146 | 0G | 0GUSDT (spot) | spot, usdm-futures | 94.5% | 0.068 | BABABUSDT | 100.0% | 0.8 | 912.6K | 181.3% |
| 147 | TST | TSTUSDT (spot) | spot, usdm-futures | 94.4% | 0.082 | MAGMAUSDT | 100.0% | 0.3 | 309.9K | 97.8% |
| 148 | NIGHT | NIGHTUSDT (spot) | spot, usdm-futures | 94.4% | 0.101 | BTCUSDT | 100.0% | 0.1 | 8.1M | 220.9% |
| 149 | TREE | TREEUSDT (spot) | spot, usdm-futures | 94.4% | 0.097 | JSTUSDT | 100.0% | 0.5 | 2.3M | 200.2% |
| 150 | BANANA | BANANAUSDT (spot) | spot, usdm-futures | 94.3% | 0.077 | BTCUSDT | 100.0% | 0.2 | 737.4K | 108.9% |
| 151 | M | MUSDT (usdm-futures) | usdm-futures | 94.3% | 0.116 | BTCUSDT | 100.0% | 0.1 | 2.1M | 78.9% |
| 152 | SENT | SENTUSDT (spot) | spot, usdm-futures | 94.2% | 0.072 | HUSDT | 100.0% | 0.4 | 591.8K | 57.3% |
| 153 | MTL | MTLUSDT (spot) | spot, usdm-futures | 94.2% | 0.088 | BOTUSDT | 100.0% | 2.7 | 31.6K | 57.5% |
| 154 | ON | ONUSDT (usdm-futures) | usdm-futures | 94.1% | 0.068 | AIAUSDT | 100.0% | 0 | 27.7M | 363.8% |
| 155 | MET | METUSDT (spot) | spot, usdm-futures | 94.0% | 0.096 | BTCUSDT | 100.0% | 0.2 | 759.7K | 119.0% |
| 156 | TA | TAUSDT (usdm-futures) | usdm-futures | 94.0% | 0.084 | AIAUSDT | 100.0% | 0.1 | 4.6M | 137.9% |
| 157 | HFT | HFTUSDT (spot) | spot, usdm-futures | 94.0% | 0.097 | BTCUSDT | 100.0% | 1.3 | 383.8K | 131.5% |
| 158 | POWER | POWERUSDT (usdm-futures) | usdm-futures | 94.0% | 0.120 | BTCUSDT | 100.0% | 0 | 2.3M | 78.0% |
| 159 | OSMO | OSMOUSDT (spot) | spot | 93.9% | 0.072 | JUVUSDT | 100.0% | 0.9 | 104.6K | 118.9% |
| 160 | BMT | BMTUSDT (spot) | spot, usdm-futures | 93.7% | 0.149 | BTCUSDT | 100.0% | 0.5 | 566.6K | 113.4% |
| 161 | RECALL | RECALLUSDT (usdm-futures) | usdm-futures | 93.7% | 0.110 | BTCUSDT | 100.0% | 0.1 | 2.1M | 141.9% |
| 162 | SKL | SKLUSDT (spot) | spot, usdm-futures | 93.7% | 0.129 | FFUSDT | 100.0% | 0.6 | 730.4K | 100.2% |
| 163 | HYPER | HYPERUSDT (spot) | spot, usdm-futures | 93.7% | 0.090 | HDUSDT | 100.0% | 0.6 | 332.7K | 67.3% |
| 164 | SOMI | SOMIUSDT (spot) | spot, usdm-futures | 93.6% | 0.109 | BTCUSDT | 100.0% | 0.4 | 261.8K | 70.4% |
| 165 | TNSR | TNSRUSDT (spot) | spot, usdm-futures | 93.5% | 0.122 | BTCUSDT | 100.0% | 0.6 | 290K | 89.4% |
| 166 | SNOW | SNOWUSDT (usdm-futures) | usdm-futures | 93.5% | 0.096 | NVOUSDT | 100.0% | 0.1 | 371.4K | 81.9% |
| 167 | BRKB | BRKBUSDT (usdm-futures) | usdm-futures | 93.4% | 0.076 | 币安人生USDT | 100.0% | 0.1 | 1.3M | 32.2% |
| 168 | MMT | MMTUSDT (spot) | spot, usdm-futures | 93.4% | 0.087 | BTCUSDT | 100.0% | 0.3 | 522.2K | 81.9% |
| 169 | PRL | PRLUSDT (usdm-futures) | usdm-futures | 93.4% | 0.098 | NMRUSDT | 100.0% | 0.1 | 681.2K | 69.7% |
| 170 | AUCTION | AUCTIONUSDT (spot) | spot, usdm-futures | 93.3% | 0.100 | WETUSDT | 100.0% | 0.5 | 485.8K | 47.0% |
| 171 | B2 | B2USDT (usdm-futures) | usdm-futures | 93.3% | 0.197 | AWEUSDT | 100.0% | 0.1 | 56.6M | 858.8% |
| 172 | SPACE | SPACEUSDT (usdm-futures) | usdm-futures | 93.2% | 0.130 | BTCUSDT | 100.0% | 0.1 | 2.5M | 73.4% |
| 173 | GENIUS | GENIUSUSDT (spot) | spot, usdm-futures | 93.2% | 0.084 | PARTIUSDT | 100.0% | 0.1 | 835.7K | 108.5% |
| 174 | FIGHT | FIGHTUSDT (usdm-futures) | usdm-futures | 93.2% | 0.089 | KGSTUSDT | 100.0% | 0.1 | 1.4M | 126.2% |
| 175 | NAORIS | NAORISUSDT (usdm-futures) | usdm-futures | 93.1% | 0.076 | FFUSDT | 100.0% | 0.1 | 8.1M | 414.4% |
| 176 | TWT | TWTUSDT (spot) | spot, usdm-futures | 93.1% | 0.104 | BTCUSDT | 100.0% | 0.5 | 250.3K | 46.3% |
| 177 | TUT | TUTUSDT (spot) | spot, usdm-futures | 92.9% | 0.081 | DODOUSDT | 100.0% | 0.2 | 932.3K | 133.2% |
| 178 | MYX | MYXUSDT (usdm-futures) | usdm-futures | 92.9% | 0.132 | BTCUSDT | 100.0% | 0 | 5.3M | 133.5% |
| 179 | KOMA | KOMAUSDT (usdm-futures) | usdm-futures | 92.9% | 0.095 | CROSSUSDT | 100.0% | 0.1 | 1.1M | 112.1% |
| 180 | ZKP | ZKPUSDT (spot) | spot, usdm-futures | 92.9% | 0.118 | BTCUSDT | 100.0% | 0.8 | 344.2K | 87.5% |
| 181 | LSK | LSKUSDT (spot) | spot, usdm-futures | 92.8% | 0.081 | BMTUSDT | 100.0% | 1.1 | 34.1K | 46.1% |
| 182 | 龙虾 | 龙虾USDT (usdm-futures) | usdm-futures | 92.8% | 0.109 | BTCUSDT | 100.0% | 0 | 2.1M | 132.1% |
| 183 | LLY | LLYUSDT (usdm-futures) | usdm-futures | 92.7% | 0.086 | VANRYUSDT | 100.0% | 0.1 | 999.3K | 37.7% |
| 184 | MANTA | MANTAUSDT (spot) | spot, usdm-futures | 92.7% | 0.104 | BTCUSDT | 100.0% | 0.2 | 200.4K | 59.3% |
| 185 | MUBARAK | MUBARAKUSDT (spot) | spot, usdm-futures | 92.6% | 0.121 | BTCUSDT | 100.0% | 0.3 | 403.1K | 78.2% |
| 186 | KSM | KSMUSDT (spot) | spot, usdm-futures | 92.5% | 0.114 | BTCUSDT | 100.0% | 0.7 | 172.7K | 91.5% |
| 187 | AIN | AINUSDT (usdm-futures) | usdm-futures | 92.5% | 0.095 | JCTUSDT | 100.0% | 0 | 985.1K | 144.7% |
| 188 | PROM | PROMUSDT (spot) | spot, usdm-futures | 92.5% | 0.113 | TACUSDT | 100.0% | 0.1 | 2.6M | 302.5% |
| 189 | CGPT | CGPTUSDT (spot) | spot, usdm-futures | 92.4% | 0.084 | MYXUSDT | 100.0% | 0.3 | 382.7K | 83.2% |
| 190 | BULLA | BULLAUSDT (usdm-futures) | usdm-futures | 92.3% | 0.109 | OSMOUSDT | 100.0% | 0 | 8M | 275.4% |
| 191 | CBRS | CBRSBUSDT (spot) | spot, usdm-futures | 92.3% | 0.110 | HDUSDT | 100.0% | 0.1 | 800.6K | 229.1% |
| 192 | AAPL | AAPLUSDT (usdm-futures) | usdm-futures | 92.3% | 0.106 | BSPUSDT | 100.0% | 0.1 | 18.5M | 30.0% |
| 193 | AUDIO | AUDIOUSDT (spot) | spot | 92.3% | 0.127 | BTCUSDT | 100.0% | 0.3 | 221.9K | 60.8% |
| 194 | SUN | SUNUSDT (spot) | spot, usdm-futures | 92.2% | 0.110 | ATMUSDT | 100.0% | 1.2 | 327.2K | 17.6% |
| 195 | ARIA | ARIAUSDT (usdm-futures) | usdm-futures | 92.1% | 0.158 | BTCUSDT | 100.0% | 0.1 | 5M | 189.6% |
| 196 | BAND | BANDUSDT (spot) | spot, usdm-futures | 92.1% | 0.097 | LISTAUSDT | 100.0% | 0.6 | 34.1K | 44.4% |
| 197 | KAVA | KAVAUSDT (spot) | spot, usdm-futures | 92.1% | 0.082 | BRKBUSDT | 100.0% | 0.3 | 198.4K | 37.0% |
| 198 | FTT | FTTUSDT (spot) | spot | 92.0% | 0.073 | XECUSDT | 100.0% | 0.3 | 166.3K | 155.3% |
| 199 | LUNA | LUNAUSDT (spot) | spot | 92.0% | 0.112 | BTCUSDT | 100.0% | 1 | 196.5K | 81.5% |
| 200 | ALLO | ALLOUSDT (spot) | spot, usdm-futures | 91.9% | 0.086 | SONYUSDT | 100.0% | 0 | 4M | 117.9% |
| 201 | LAB | LABUSDT (usdm-futures) | usdm-futures | 91.8% | 0.142 | BEATUSDT | 100.0% | 0.1 | 227.6M | 518.2% |
| 202 | QNTX | QNTXUSDT (usdm-futures) | usdm-futures | 91.8% | 0.100 | SNOWUSDT | 100.0% | 0.1 | 8.2M | 128.1% |
| 203 | YB | YBUSDT (spot) | spot, usdm-futures | 91.8% | 0.109 | SYNUSDT | 100.0% | 0.5 | 274.5K | 91.2% |
| 204 | IQ | IQUSDT (spot) | spot | 91.8% | 0.091 | BSPUSDT | 100.0% | 1.3 | 26.4K | 50.4% |
| 205 | SHIB | SHIBUSDT (spot) | spot, usdm-futures | 91.7% | 0.171 | BTCUSDT | 100.0% | 0.3 | 1.9M | 108.9% |
| 206 | SKR | SKRUSDT (usdm-futures) | usdm-futures | 91.7% | 0.085 | WETUSDT | 100.0% | 0.1 | 1.2M | 71.4% |
| 207 | SAPIEN | SAPIENUSDT (spot) | spot, usdm-futures | 91.6% | 0.131 | BTCUSDT | 100.0% | 0.4 | 1.8M | 142.1% |
| 208 | FRAX | FRAXUSDT (spot) | spot, usdm-futures | 91.6% | 0.111 | BTCUSDT | 100.0% | 0.4 | 91.4K | 53.4% |
| 209 | AGLD | AGLDUSDT (spot) | spot, usdm-futures | 91.5% | 0.117 | SOMIUSDT | 100.0% | 0.3 | 245.8K | 84.1% |
| 210 | BILL | BILLUSDT (usdm-futures) | usdm-futures | 91.4% | 0.101 | TRIAUSDT | 100.0% | 0.1 | 11.9M | 167.9% |
| 211 | ESPORTS | ESPORTSUSDT (usdm-futures) | usdm-futures | 91.4% | 0.086 | TACUSDT | 100.0% | 0 | 86M | 315.7% |
| 212 | IWM | IWMUSDT (usdm-futures) | usdm-futures | 91.3% | 0.095 | HIMSUSDT | 100.0% | 0.2 | 237.3K | 22.2% |
| 213 | GNS | GNSUSDT (spot) | spot | 91.3% | 0.090 | STRAXUSDT | 100.0% | 0.5 | 46.2K | 61.2% |
| 214 | ACE | ACEUSDT (spot) | spot, usdm-futures | 91.2% | 0.078 | BTWUSDT | 100.0% | 0.1 | 4.4M | 270.6% |
| 215 | GLMR | GLMRUSDT (spot) | spot | 91.1% | 0.107 | POLYXUSDT | 100.0% | 0.2 | 878.5K | 242.8% |
| 216 | NEXO | NEXOUSDT (spot) | spot | 91.1% | 0.132 | BTCUSDT | 100.0% | 0.3 | 584.5K | 79.4% |
| 217 | TRUST | TRUSTUSDT (usdm-futures) | usdm-futures | 91.1% | 0.113 | BTCUSDT | 100.0% | 0.1 | 1.6M | 91.3% |
| 218 | GWEI | GWEIUSDT (usdm-futures) | usdm-futures | 91.0% | 0.088 | BROCCOLIF3BUSDT | 100.0% | 0.1 | 9.9M | 201.9% |
| 219 | GEV | GEVUSDT (usdm-futures) | usdm-futures | 91.0% | 0.131 | AKEUSDT | 100.0% | 0.1 | 5.3M | 204.9% |
| 220 | AMP | AMPUSDT (spot) | spot | 90.9% | 0.081 | JUVUSDT | 100.0% | 0.5 | 513K | 119.2% |
| 221 | SPK | SPKUSDT (spot) | spot, usdm-futures | 90.8% | 0.118 | BTCUSDT | 100.0% | 0.4 | 455.2K | 41.9% |
| 222 | LUMIA | LUMIAUSDT (spot) | spot, usdm-futures | 90.8% | 0.104 | STGUSDT | 100.0% | 0.2 | 900.5K | 110.3% |
| 223 | SQD | SQDUSDT (usdm-futures) | usdm-futures | 90.7% | 0.099 | BTCUSDT | 100.0% | 0.1 | 1.3M | 112.9% |
| 224 | CTR | CTRUSDT (usdm-futures) | usdm-futures | 90.7% | 0.095 | BTCUSDT | 100.0% | 0.1 | 541.2K | 94.7% |
| 225 | PEPE | PEPEUSDT (spot) | spot, usdm-futures | 90.6% | 0.149 | BTCUSDT | 100.0% | 0.2 | 13.5M | 174.4% |
| 226 | FOGO | FOGOUSDT (spot) | spot, usdm-futures | 90.6% | 0.091 | ACXUSDT | 100.0% | 0.6 | 134.2K | 60.7% |
| 227 | KERNEL | KERNELUSDT (spot) | spot, usdm-futures | 90.5% | 0.123 | BTCUSDT | 100.0% | 0.7 | 513.4K | 139.0% |
| 228 | PAYP | PAYPUSDT (usdm-futures) | usdm-futures | 90.5% | 0.124 | HIMSUSDT | 100.0% | 0.2 | 665.8K | 104.5% |
| 229 | NEWT | NEWTUSDT (spot) | spot, usdm-futures | 90.4% | 0.110 | BTCUSDT | 100.0% | 1.3 | 471K | 87.4% |
| 230 | BAT | BATUSDT (spot) | spot, usdm-futures | 90.4% | 0.126 | BTCUSDT | 100.0% | 1.1 | 168.6K | 47.9% |
| 231 | ZHIPU | ZHIPUUSDT (usdm-futures) | usdm-futures | 90.2% | 0.118 | BABABUSDT | 100.0% | 0 | 102.1M | 270.9% |
| 232 | SANTOS | SANTOSUSDT (spot) | spot, usdm-futures | 90.2% | 0.085 | HMSTRUSDT | 100.0% | 0.7 | 93K | 59.7% |
| 233 | CATI | CATIUSDT (spot) | spot, usdm-futures | 90.2% | 0.097 | NAORISUSDT | 100.0% | 0.3 | 169.1K | 59.7% |
| 234 | JASMY | JASMYUSDT (spot) | spot, usdm-futures | 90.2% | 0.141 | ICXUSDT | 100.0% | 0.4 | 764.8K | 84.4% |
| 235 | KGEN | KGENUSDT (usdm-futures) | usdm-futures | 90.1% | 0.096 | GPSUSDT | 100.0% | 0.1 | 1.6M | 109.1% |
| 236 | ASR | ASRUSDT (spot) | spot, usdm-futures | 90.0% | 0.138 | BTCUSDT | 100.0% | 0.7 | 109.3K | 37.8% |
| 237 | WOO | WOOUSDT (spot) | spot, usdm-futures | 89.9% | 0.109 | IWMUSDT | 100.0% | 0.5 | 98.8K | 69.9% |
| 238 | BLUR | BLURUSDT (spot) | spot, usdm-futures | 89.9% | 0.092 | BTCUSDT | 100.0% | 0.3 | 2.6M | 130.8% |
| 239 | HUMA | HUMAUSDT (spot) | spot, usdm-futures | 89.8% | 0.138 | ENSOUSDT | 100.0% | 0.3 | 311.7K | 68.5% |
| 240 | HIVE | HIVEUSDT (spot) | spot, usdm-futures | 89.8% | 0.104 | 1000CHEEMSUSDT | 100.0% | 0.9 | 77.1K | 54.4% |
| 241 | MBL | MBLUSDT (spot) | spot | 89.8% | 0.089 | AUCTIONUSDT | 100.0% | 0.2 | 312.1K | 125.5% |
| 242 | SLX | SLXUSDT (usdm-futures) | usdm-futures | 89.6% | 0.141 | BTCUSDT | 100.0% | 0 | 21.5M | 145.7% |
| 243 | ACT | ACTUSDT (spot) | spot, usdm-futures | 89.6% | 0.189 | BTCUSDT | 100.0% | 0.2 | 529.3K | 102.0% |
| 244 | CYBER | CYBERUSDT (spot) | spot, usdm-futures | 89.5% | 0.099 | KSMUSDT | 100.0% | 1.7 | 153.9K | 66.6% |
| 245 | RPL | RPLUSDT (spot) | spot, usdm-futures | 89.5% | 0.074 | ONEUSDT | 100.0% | 1.2 | 167.7K | 155.5% |
| 246 | AGT | AGTUSDT (usdm-futures) | usdm-futures | 89.5% | 0.093 | BRKBUSDT | 100.0% | 0 | 2.9M | 204.1% |
| 247 | GUN | GUNUSDT (spot) | spot, usdm-futures | 89.4% | 0.136 | SPKUSDT | 100.0% | 1 | 1.5M | 153.0% |
| 248 | GMX | GMXUSDT (spot) | spot, usdm-futures | 89.3% | 0.134 | BTCUSDT | 100.0% | 1.2 | 395.3K | 95.1% |
| 249 | ESP | ESPUSDT (spot) | spot, usdm-futures | 89.2% | 0.087 | ACTUSDT | 100.0% | 0.3 | 197.1K | 52.7% |
| 250 | QI | QIUSDT (spot) | spot | 89.2% | 0.143 | ATMUSDT | 100.0% | 0.6 | 214.8K | 165.3% |
| 251 | RKLB | RKLBBUSDT (spot) | spot, usdm-futures | 89.2% | 0.113 | FRAXUSDT | 100.0% | 0.1 | 226.3K | 151.1% |
| 252 | VANA | VANAUSDT (spot) | spot, usdm-futures | 89.1% | 0.104 | MYXUSDT | 100.0% | 0.2 | 8.2M | 84.2% |
| 253 | EDU | EDUUSDT (spot) | spot, usdm-futures | 89.1% | 0.198 | BTCUSDT | 100.0% | 2.4 | 301.9K | 80.9% |
| 254 | ACU | ACUUSDT (usdm-futures) | usdm-futures | 89.0% | 0.162 | BTCUSDT | 100.0% | 0.1 | 556.3K | 45.6% |
| 255 | MANTRA | MANTRAUSDT (spot) | spot, usdm-futures | 89.0% | 0.134 | BTCUSDT | 100.0% | 0.4 | 631.7K | 86.7% |
| 256 | STRC | STRCUSDT (usdm-futures) | usdm-futures | 88.9% | 0.088 | HIMSUSDT | 100.0% | 0.1 | 1.5M | 25.8% |
| 257 | TXN | TXNUSDT (usdm-futures) | usdm-futures | 88.9% | 0.146 | TOWNSUSDT | 100.0% | 0.1 | 3.6M | 229.6% |
| 258 | NIL | NILUSDT (spot) | spot, usdm-futures | 88.9% | 0.130 | BTCUSDT | 100.0% | 0.2 | 531.4K | 96.0% |
| 259 | PORTAL | PORTALUSDT (spot) | spot, usdm-futures | 88.8% | 0.111 | BTCUSDT | 100.0% | 0.2 | 403.3K | 69.6% |
| 260 | INX | INXUSDT (usdm-futures) | usdm-futures | 88.7% | 0.179 | BTCUSDT | 100.0% | 0.1 | 1.5M | 104.4% |
| 261 | WAXP | WAXPUSDT (spot) | spot, usdm-futures | 88.7% | 0.090 | KSMUSDT | 100.0% | 1 | 52.6K | 58.9% |
| 262 | ICNT | ICNTUSDT (usdm-futures) | usdm-futures | 88.6% | 0.129 | TREEUSDT | 100.0% | 0.1 | 2.4M | 158.1% |
| 263 | TRADOOR | TRADOORUSDT (usdm-futures) | usdm-futures | 88.5% | 0.129 | BTCUSDT | 100.0% | 0 | 3.2M | 115.7% |
| 264 | XVS | XVSUSDT (spot) | spot, usdm-futures | 88.5% | 0.100 | TWTUSDT | 100.0% | 1.1 | 55.8K | 84.3% |
| 265 | DOOD | DOODUSDT (usdm-futures) | usdm-futures | 88.5% | 0.204 | BTCUSDT | 100.0% | 0.2 | 1.6M | 96.7% |
| 266 | HOLO | HOLOUSDT (spot) | spot, usdm-futures | 88.4% | 0.082 | INITUSDT | 100.0% | 0.3 | 1.8M | 137.0% |
| 267 | METIS | METISUSDT (spot) | spot, usdm-futures | 88.4% | 0.129 | WOOUSDT | 100.0% | 1.8 | 98.8K | 71.7% |
| 268 | SOFI | SOFIUSDT (usdm-futures) | usdm-futures | 88.3% | 0.102 | BTCUSDT | 100.0% | 0.1 | 1.1M | 97.0% |
| 269 | G | GUSDT (spot) | spot, usdm-futures | 88.2% | 0.094 | METUSDT | 100.0% | 1.8 | 365.8K | 73.5% |
| 270 | SAFE | SAFEUSDT (usdm-futures) | usdm-futures | 88.1% | 0.098 | MEUSDT | 100.0% | 0.1 | 3.3M | 143.1% |
| 271 | IDOL | IDOLUSDT (usdm-futures) | usdm-futures | 88.0% | 0.101 | BTCUSDT | 100.0% | 0.2 | 856.5K | 81.5% |
| 272 | APR | APRUSDT (usdm-futures) | usdm-futures | 88.0% | 0.118 | BTCUSDT | 100.0% | 0.1 | 1.8M | 77.8% |
| 273 | AT | ATUSDT (spot) | spot, usdm-futures | 87.9% | 0.130 | EDUUSDT | 100.0% | 0.4 | 132.4K | 52.6% |
| 274 | GTC | GTCUSDT (spot) | spot, usdm-futures | 87.9% | 0.126 | BTCUSDT | 100.0% | 1.2 | 119.8K | 105.1% |
| 275 | WMT | WMTUSDT (usdm-futures) | usdm-futures | 87.9% | 0.108 | QNTXUSDT | 100.0% | 0.2 | 517.9K | 32.1% |
| 276 | QTUM | QTUMUSDT (spot) | spot, usdm-futures | 87.8% | 0.138 | BTCUSDT | 100.0% | 0.9 | 133.1K | 55.2% |
| 277 | STEEM | STEEMUSDT (spot) | spot, usdm-futures | 87.8% | 0.099 | MUBARAKUSDT | 100.0% | 0.9 | 36.1K | 34.9% |
| 278 | MOVE | MOVEUSDT (spot) | spot, usdm-futures | 87.7% | 0.118 | EPICUSDT | 100.0% | 3.2 | 372.6K | 107.2% |
| 279 | UB | UBUSDT (usdm-futures) | usdm-futures | 87.7% | 0.099 | JCTUSDT | 100.0% | 0 | 35.8M | 214.7% |
| 280 | DATAIP | DATAIPUSDT (usdm-futures) | usdm-futures | 87.6% | 0.095 | EGLDUSDT | 100.0% | 0.1 | 895.6K | 48.9% |
| 281 | XLE | XLEUSDT (usdm-futures) | usdm-futures | 87.6% | 0.112 | HDUSDT | 100.0% | 0.1 | 365.9K | 33.1% |
| 282 | GUA | GUAUSDT (usdm-futures) | usdm-futures | 87.5% | 0.160 | PLAYUSDT | 100.0% | 0.1 | 1.4M | 90.7% |
| 283 | XPT | XPTUSDT (usdm-futures) | usdm-futures | 87.5% | 0.176 | BTCUSDT | 100.0% | 0 | 7.4M | 43.6% |
| 284 | ADX | ADXUSDT (spot) | spot | 87.5% | 0.100 | GNSUSDT | 100.0% | 0.3 | 222.6K | 79.0% |
| 285 | BSB | BSBUSDT (usdm-futures) | usdm-futures | 87.3% | 0.166 | BTCUSDT | 100.0% | 0 | 15M | 162.2% |
| 286 | PSG | PSGUSDT (spot) | spot | 87.3% | 0.102 | TFUELUSDT | 100.0% | 0.4 | 194.4K | 104.2% |
| 287 | BX | BXUSDT (usdm-futures) | usdm-futures | 87.3% | 0.166 | JPMUSDT | 100.0% | 0.1 | 2M | 58.9% |
| 288 | USTC | USTCUSDT (spot) | spot, usdm-futures | 87.2% | 0.110 | MYXUSDT | 100.0% | 0.8 | 151.6K | 65.3% |
| 289 | HEMI | HEMIUSDT (spot) | spot, usdm-futures | 87.2% | 0.117 | BTCUSDT | 100.0% | 0.2 | 2.7M | 195.1% |
| 290 | NOT | NOTUSDT (spot) | spot, usdm-futures | 87.1% | 0.130 | BMTUSDT | 100.0% | 0.8 | 165.9K | 80.3% |
| 291 | ENS | ENSUSDT (spot) | spot, usdm-futures | 87.0% | 0.156 | BTCUSDT | 100.0% | 0.3 | 1.5M | 99.2% |
| 292 | AXL | AXLUSDT (spot) | spot, usdm-futures | 87.0% | 0.105 | MANTAUSDT | 100.0% | 0.6 | 293.6K | 89.4% |
| 293 | SKYAI | SKYAIUSDT (usdm-futures) | usdm-futures | 87.0% | 0.130 | BTCUSDT | 100.0% | 0.1 | 4.8M | 131.5% |
| 294 | SOLV | SOLVUSDT (spot) | spot, usdm-futures | 86.9% | 0.121 | ICXUSDT | 100.0% | 1.4 | 285.2K | 121.5% |
| 295 | BASED | BASEDUSDT (usdm-futures) | usdm-futures | 86.9% | 0.217 | BTCUSDT | 100.0% | 0 | 6.9M | 97.0% |
| 296 | ZKC | ZKCUSDT (spot) | spot, usdm-futures | 86.8% | 0.097 | 1000CHEEMSUSDT | 100.0% | 0.7 | 991.2K | 103.9% |
| 297 | ACH | ACHUSDT (spot) | spot, usdm-futures | 86.8% | 0.161 | BTCUSDT | 100.0% | 1 | 203.7K | 81.7% |
| 298 | RESOLV | RESOLVUSDT (spot) | spot, usdm-futures | 86.7% | 0.106 | BTCUSDT | 100.0% | 0.7 | 896.4K | 106.4% |
| 299 | KAT | KATUSDT (spot) | spot, usdm-futures | 86.7% | 0.115 | BATUSDT | 100.0% | 0.6 | 293.3K | 70.4% |
| 300 | U | UUSDT (spot) | spot | 86.6% | 0.080 | BTCUSDT | 100.0% | 0.3 | 15.7M | 4.8% |
| 301 | AAOI | AAOIBUSDT (spot) | spot, usdm-futures | 86.6% | 0.107 | BTCUSDT | 100.0% | 0.1 | 217.8K | 208.5% |
| 302 | PIEVERSE | PIEVERSEUSDT (usdm-futures) | usdm-futures | 86.6% | 0.176 | BTCUSDT | 100.0% | 0.1 | 3.6M | 98.8% |
| 303 | EBAY | EBAYUSDT (usdm-futures) | usdm-futures | 86.4% | 0.137 | AAPLUSDT | 100.0% | 0.1 | 440.8K | 46.0% |
| 304 | GNO | GNOUSDT (spot) | spot | 86.4% | 0.094 | NEWTUSDT | 100.0% | 0.2 | 121.3K | 63.1% |
| 305 | BANANAS31 | BANANAS31USDT (spot) | spot, usdm-futures | 86.3% | 0.171 | BTCUSDT | 100.0% | 0.2 | 284.8K | 59.9% |
| 306 | CTSI | CTSIUSDT (spot) | spot, usdm-futures | 86.3% | 0.126 | ENSUSDT | 100.0% | 1 | 95.6K | 45.9% |
| 307 | EWZ | EWZUSDT (usdm-futures) | usdm-futures | 86.2% | 0.118 | LLYUSDT | 100.0% | 0.2 | 205.5K | 33.6% |
| 308 | TAIKO | TAIKOUSDT (usdm-futures) | usdm-futures | 86.2% | 0.167 | BTCUSDT | 100.0% | 0.1 | 2M | 88.9% |
| 309 | ENJ | ENJUSDT (spot) | spot, usdm-futures | 86.1% | 0.175 | BTCUSDT | 100.0% | 0.2 | 244.4K | 50.7% |
| 310 | PIVX | PIVXUSDT (spot) | spot | 86.1% | 0.130 | PAYPUSDT | 100.0% | 0.4 | 662.7K | 290.8% |
| 311 | QNT | QNTUSDT (spot) | spot, usdm-futures | 86.0% | 0.173 | BTCUSDT | 100.0% | 0.6 | 406.2K | 42.3% |
| 312 | ARC | ARCUSDT (usdm-futures) | usdm-futures | 85.9% | 0.142 | BTCUSDT | 100.0% | 0.1 | 1.1M | 72.6% |
| 313 | UMA | UMAUSDT (spot) | spot, usdm-futures | 85.9% | 0.129 | METISUSDT | 100.0% | 1.5 | 66.6K | 49.9% |
| 314 | ASTER | ASTERUSDT (spot) | spot, usdm-futures | 85.8% | 0.105 | BTCUSDT | 100.0% | 0.3 | 2.5M | 67.2% |
| 315 | US | USUSDT (usdm-futures) | usdm-futures | 85.8% | 0.111 | TRIAUSDT | 100.0% | 0 | 25.7M | 199.4% |
| 316 | C98 | C98USDT (spot) | spot, usdm-futures | 85.7% | 0.121 | AGLDUSDT | 100.0% | 0.6 | 122.9K | 56.9% |
| 317 | VTHO | VTHOUSDT (spot) | spot, usdm-futures | 85.6% | 0.127 | BTCUSDT | 100.0% | 1.3 | 141.2K | 60.7% |
| 318 | CC | CCUSDT (usdm-futures) | usdm-futures | 85.6% | 0.182 | BTCUSDT | 100.0% | 0.1 | 2.5M | 42.1% |
| 319 | FLNC | FLNCUSDT (usdm-futures) | usdm-futures | 85.5% | 0.169 | BTCUSDT | 100.0% | 0.2 | 5M | 144.9% |
| 320 | MAVIA | MAVIAUSDT (usdm-futures) | usdm-futures | 85.5% | 0.162 | BTCUSDT | 100.0% | 0.1 | 414.1K | 63.8% |
| 321 | QKC | QKCUSDT (spot) | spot | 85.5% | 0.094 | GLMRUSDT | 100.0% | 0.3 | 40.1K | 88.2% |
| 322 | LA | LAUSDT (spot) | spot, usdm-futures | 85.4% | 0.189 | ERAUSDT | 100.0% | 0.2 | 3.2M | 204.5% |
| 323 | POWR | POWRUSDT (spot) | spot, usdm-futures | 85.4% | 0.123 | BANANAS31USDT | 100.0% | 0.6 | 67.2K | 70.7% |
| 324 | OG | OGUSDT (spot) | spot, usdm-futures | 85.4% | 0.115 | SCRTUSDT | 100.0% | 0.4 | 209.9K | 49.5% |
| 325 | GME | GMEUSDT (usdm-futures) | usdm-futures | 85.2% | 0.112 | BLURUSDT | 100.0% | 0.1 | 348.2K | 30.9% |
| 326 | RVN | RVNUSDT (spot) | spot, usdm-futures | 85.2% | 0.165 | BTCUSDT | 100.0% | 0.8 | 159.2K | 76.3% |
| 327 | VIC | VICUSDT (spot) | spot, usdm-futures | 85.2% | 0.115 | CBRSBUSDT | 100.0% | 0.6 | 390.3K | 110.4% |
| 328 | BTR | BTRUSDT (usdm-futures) | usdm-futures | 85.1% | 0.171 | BTCUSDT | 100.0% | 0.1 | 1M | 93.3% |
| 329 | ELSA | ELSAUSDT (usdm-futures) | usdm-futures | 85.1% | 0.137 | BTCUSDT | 100.0% | 0.1 | 3.9M | 79.3% |
| 330 | PHAROS | PHAROSUSDT (usdm-futures) | usdm-futures | 85.0% | 0.093 | BATUSDT | 100.0% | 0 | 3.5M | 139.5% |
| 331 | UAI | UAIUSDT (usdm-futures) | usdm-futures | 85.0% | 0.099 | JASMYUSDT | 100.0% | 0.1 | 8.3M | 148.1% |
| 332 | BIO | BIOUSDT (spot) | spot, usdm-futures | 84.9% | 0.213 | BTCUSDT | 100.0% | 0.1 | 2.1M | 105.2% |
| 333 | XVG | XVGUSDT (spot) | spot, usdm-futures | 84.9% | 0.128 | NOTUSDT | 100.0% | 0.7 | 115.1K | 67.7% |
| 334 | ALPINE | ALPINEUSDT (spot) | spot, usdm-futures | 84.7% | 0.120 | AINUSDT | 100.0% | 1.1 | 122.8K | 80.7% |
| 335 | SIREN | SIRENUSDT (usdm-futures) | usdm-futures | 84.7% | 0.164 | BTCUSDT | 100.0% | 0.1 | 10.9M | 176.4% |
| 336 | CELO | CELOUSDT (spot) | spot, usdm-futures | 84.6% | 0.175 | BTCUSDT | 100.0% | 0.2 | 482.5K | 77.0% |
| 337 | JTO | JTOUSDT (spot) | spot, usdm-futures | 84.6% | 0.163 | BTCUSDT | 100.0% | 0.1 | 2.5M | 93.1% |
| 338 | SHELL | SHELLUSDT (spot) | spot, usdm-futures | 84.5% | 0.116 | B2USDT | 100.0% | 0.8 | 933.2K | 159.5% |
| 339 | TLM | TLMUSDT (spot) | spot, usdm-futures | 84.4% | 0.105 | BLURUSDT | 100.0% | 0.1 | 5.5M | 346.8% |
| 340 | PROVE | PROVEUSDT (spot) | spot, usdm-futures | 84.3% | 0.138 | BTCUSDT | 100.0% | 0.4 | 225.2K | 53.4% |
| 341 | SSV | SSVUSDT (spot) | spot, usdm-futures | 84.3% | 0.148 | XVGUSDT | 100.0% | 0.7 | 355.7K | 60.2% |
| 342 | LQTY | LQTYUSDT (spot) | spot, usdm-futures | 84.2% | 0.112 | WAXPUSDT | 100.0% | 1.1 | 150.6K | 85.8% |
| 343 | BIRB | BIRBUSDT (usdm-futures) | usdm-futures | 84.2% | 0.192 | BTCUSDT | 100.0% | 0.1 | 1.5M | 85.4% |
| 344 | BAR | BARUSDT (spot) | spot | 84.1% | 0.143 | CITYUSDT | 100.0% | 0.5 | 469.9K | 160.8% |
| 345 | FLEX | FLEXUSDT (usdm-futures) | usdm-futures | 84.1% | 0.136 | LLYUSDT | 100.0% | 0.1 | 266.3K | 72.3% |
| 346 | MASK | MASKUSDT (spot) | spot, usdm-futures | 84.1% | 0.123 | BTCUSDT | 100.0% | 1.7 | 88.4K | 53.5% |
| 347 | PEOPLE | PEOPLEUSDT (spot) | spot, usdm-futures | 84.0% | 0.181 | BTCUSDT | 100.0% | 1.1 | 95K | 55.1% |
| 348 | BANK | BANKUSDT (spot) | spot, usdm-futures | 83.9% | 0.121 | MYXUSDT | 100.0% | 0 | 97.1M | 761.0% |
| 349 | CLO | CLOUSDT (usdm-futures) | usdm-futures | 83.9% | 0.092 | WETUSDT | 100.0% | 0 | 9.2M | 212.6% |
| 350 | XMR | XMRUSDT (usdm-futures) | usdm-futures | 83.8% | 0.260 | BTCUSDT | 100.0% | 0.1 | 29.8M | 56.9% |
| 351 | FLOW | FLOWUSDT (spot) | spot, usdm-futures | 83.8% | 0.143 | BTCUSDT | 100.0% | 0.5 | 127.9K | 45.0% |
| 352 | FHE | FHEUSDT (usdm-futures) | usdm-futures | 83.7% | 0.166 | BTCUSDT | 100.0% | 0.1 | 1.2M | 107.7% |
| 353 | SCR | SCRUSDT (spot) | spot, usdm-futures | 83.7% | 0.144 | BTCUSDT | 100.0% | 0.3 | 233.6K | 70.5% |
| 354 | HANA | HANAUSDT (usdm-futures) | usdm-futures | 83.5% | 0.180 | OUSDT | 100.0% | 0.1 | 7.7M | 352.3% |
| 355 | MELANIA | MELANIAUSDT (usdm-futures) | usdm-futures | 83.4% | 0.137 | KATUSDT | 100.0% | 0.1 | 740K | 60.0% |
| 356 | RONIN | RONINUSDT (spot) | spot, usdm-futures | 83.4% | 0.152 | QTUMUSDT | 100.0% | 0.4 | 219.4K | 76.1% |
| 357 | IN | INUSDT (usdm-futures) | usdm-futures | 83.4% | 0.206 | BTCUSDT | 100.0% | 0.1 | 2.3M | 71.1% |
| 358 | BICO | BICOUSDT (spot) | spot, usdm-futures | 83.3% | 0.170 | BTCUSDT | 100.0% | 0.5 | 349.6K | 59.0% |
| 359 | SC | SCUSDT (spot) | spot | 83.3% | 0.118 | QKCUSDT | 100.0% | 0.5 | 59.1K | 76.9% |
| 360 | RED | REDUSDT (spot) | spot, usdm-futures | 83.2% | 0.196 | BTCUSDT | 100.0% | 0.4 | 173K | 63.3% |
| 361 | BEL | BELUSDT (spot) | spot, usdm-futures | 83.2% | 0.121 | VTHOUSDT | 100.0% | 0.4 | 375.8K | 88.8% |
| 362 | GRAM | GRAMUSDT (spot) | spot, usdm-futures | 83.2% | 0.212 | BTCUSDT | 100.0% | 0.1 | 9.8M | 85.0% |
| 363 | BOB | 1000000BOBUSDT (usdm-futures) | usdm-futures | 83.0% | 0.131 | BTCUSDT | 100.0% | 0.2 | 603.8K | 83.9% |
| 364 | KNC | KNCUSDT (spot) | spot, usdm-futures | 83.0% | 0.168 | FLOWUSDT | 100.0% | 0.8 | 121.8K | 43.8% |
| 365 | BREV | BREVUSDT (spot) | spot, usdm-futures | 82.9% | 0.120 | NMRUSDT | 100.0% | 0.8 | 135.4K | 72.6% |
| 366 | ZEREBRO | ZEREBROUSDT (usdm-futures) | usdm-futures | 82.8% | 0.199 | BTCUSDT | 100.0% | 0 | 2.3M | 108.3% |
| 367 | PTB | PTBUSDT (usdm-futures) | usdm-futures | 82.8% | 0.107 | LLYUSDT | 100.0% | 0.1 | 2.3M | 139.3% |
| 368 | TAKE | TAKEUSDT (usdm-futures) | usdm-futures | 82.8% | 0.117 | SOLVUSDT | 100.0% | 0.1 | 816.4K | 90.9% |
| 369 | Q | QUSDT (usdm-futures) | usdm-futures | 82.7% | 0.105 | BTCUSDT | 100.0% | 0.1 | 473.7K | 60.5% |
| 370 | SPORTFUN | SPORTFUNUSDT (usdm-futures) | usdm-futures | 82.7% | 0.135 | MANTRAUSDT | 100.0% | 0.1 | 678.1K | 100.0% |
| 371 | PANW | PANWUSDT (usdm-futures) | usdm-futures | 82.7% | 0.113 | SNOWUSDT | 100.0% | 0.1 | 1M | 110.1% |
| 372 | JELLYJELLY | JELLYJELLYUSDT (usdm-futures) | usdm-futures | 82.4% | 0.171 | SNXUSDT | 100.0% | 0.1 | 3.4M | 110.7% |
| 373 | NEO | NEOUSDT (spot) | spot, usdm-futures | 82.4% | 0.225 | BTCUSDT | 100.0% | 0.4 | 487.3K | 54.2% |
| 374 | VRT | VRTUSDT (usdm-futures) | usdm-futures | 82.4% | 0.126 | GEVUSDT | 100.0% | 0.1 | 298.4K | 79.9% |
| 375 | RUNE | RUNEUSDT (spot) | spot, usdm-futures | 82.2% | 0.162 | BTCUSDT | 100.0% | 0.7 | 1.2M | 76.9% |
| 376 | ATH | ATHUSDT (usdm-futures) | usdm-futures | 82.2% | 0.203 | BTCUSDT | 100.0% | 0.1 | 2.9M | 84.2% |
| 377 | EDGE | EDGEUSDT (usdm-futures) | usdm-futures | 82.2% | 0.165 | BTCUSDT | 100.0% | 0.1 | 7.5M | 119.9% |
| 378 | REQ | REQUSDT (spot) | spot | 82.1% | 0.104 | SCUSDT | 100.0% | 0.5 | 52.8K | 117.1% |
| 379 | RLC | RLCUSDT (spot) | spot, usdm-futures | 82.0% | 0.131 | STEEMUSDT | 100.0% | 0.6 | 69.8K | 43.8% |
| 380 | STO | STOUSDT (spot) | spot, usdm-futures | 82.0% | 0.145 | MELANIAUSDT | 100.0% | 0.5 | 825.1K | 110.6% |
| 381 | PUMP | PUMPUSDT (spot) | spot, usdm-futures | 81.8% | 0.234 | BTCUSDT | 100.0% | 0.1 | 8.3M | 112.4% |
| 382 | ACM | ACMUSDT (spot) | spot | 81.8% | 0.177 | JUVUSDT | 100.0% | 0.3 | 187.5K | 147.0% |
| 383 | FORM | FORMUSDT (spot) | spot, usdm-futures | 81.8% | 0.215 | BTCUSDT | 100.0% | 0.5 | 385.4K | 68.4% |
| 384 | CHR | CHRUSDT (spot) | spot, usdm-futures | 81.8% | 0.153 | HYPERUSDT | 100.0% | 0.7 | 64.8K | 55.9% |
| 385 | OPEN | OPENUSDT (spot) | spot, usdm-futures | 81.7% | 0.174 | BTCUSDT | 100.0% | 0.3 | 422.8K | 65.2% |
| 386 | SOON | SOONUSDT (usdm-futures) | usdm-futures | 81.6% | 0.228 | BTCUSDT | 100.0% | 0.2 | 998.1K | 52.9% |
| 387 | PROMPT | PROMPTUSDT (usdm-futures) | usdm-futures | 81.5% | 0.140 | HFTUSDT | 100.0% | 0.1 | 1.6M | 102.3% |
| 388 | AVNT | AVNTUSDT (spot) | spot, usdm-futures | 81.5% | 0.205 | BTCUSDT | 100.0% | 0.3 | 269.9K | 75.4% |
| 389 | RAD | RADUSDT (spot) | spot | 81.4% | 0.086 | ZILUSDT | 100.0% | 0.4 | 125.7K | 176.1% |
| 390 | 4 | 4USDT (usdm-futures) | usdm-futures | 81.4% | 0.197 | BTCUSDT | 100.0% | 0 | 880.4K | 80.5% |
| 391 | IO | IOUSDT (spot) | spot, usdm-futures | 81.3% | 0.211 | BTCUSDT | 100.0% | 0.4 | 406.3K | 78.7% |
| 392 | FLUID | FLUIDUSDT (usdm-futures) | usdm-futures | 81.3% | 0.165 | BTCUSDT | 100.0% | 0.1 | 526.7K | 91.1% |
| 393 | SAHARA | SAHARAUSDT (spot) | spot, usdm-futures | 81.1% | 0.200 | BTCUSDT | 100.0% | 0.5 | 856.8K | 64.5% |
| 394 | BSV | BSVUSDT (usdm-futures) | usdm-futures | 81.1% | 0.202 | BTCUSDT | 100.0% | 0.2 | 698.6K | 53.8% |
| 395 | MIRA | MIRAUSDT (spot) | spot, usdm-futures | 81.0% | 0.184 | LISTAUSDT | 100.0% | 0.2 | 10.9M | 592.3% |
| 396 | FWDI | FWDIUSDT (usdm-futures) | usdm-futures | 81.0% | 0.123 | RKLBBUSDT | 100.0% | 0.1 | 2.2M | 193.8% |
| 397 | TSLA | TSLABUSDT (spot) | spot, usdm-futures | 80.9% | 0.320 | GOOGLBUSDT | 100.0% | 0 | 2.5M | 82.1% |
| 398 | ASTR | ASTRUSDT (spot) | spot, usdm-futures | 80.8% | 0.139 | AVNTUSDT | 100.0% | 0.4 | 96.5K | 49.3% |
| 399 | LAZIO | LAZIOUSDT (spot) | spot | 80.8% | 0.136 | CITYUSDT | 100.0% | 0.3 | 207K | 130.9% |
| 400 | 1INCH | 1INCHUSDT (spot) | spot, usdm-futures | 80.7% | 0.194 | BTCUSDT | 100.0% | 0.2 | 737.7K | 91.7% |
| 401 | SATS | 1000SATSUSDT (spot) | spot, usdm-futures | 80.7% | 0.134 | SOMIUSDT | 100.0% | 0.5 | 150.4K | 59.7% |
| 402 | SAGA | SAGAUSDT (spot) | spot, usdm-futures | 80.6% | 0.219 | BTCUSDT | 100.0% | 0.2 | 774.4K | 82.4% |
| 403 | MAV | MAVUSDT (spot) | spot, usdm-futures | 80.6% | 0.203 | BTCUSDT | 100.0% | 0.5 | 330.9K | 71.1% |
| 404 | FIDA | FIDAUSDT (spot) | spot, usdm-futures | 80.5% | 0.143 | KNCUSDT | 100.0% | 0.8 | 214.9K | 47.1% |
| 405 | NOK | NOKBUSDT (spot) | spot, usdm-futures | 80.5% | 0.128 | VRTUSDT | 100.0% | 0.2 | 338.2K | 112.9% |
| 406 | CHIP | CHIPUSDT (spot) | spot, usdm-futures | 80.4% | 0.160 | BTCUSDT | 100.0% | 0.1 | 1.2M | 74.9% |
| 407 | AEVO | AEVOUSDT (spot) | spot, usdm-futures | 80.3% | 0.173 | BTCUSDT | 100.0% | 0.5 | 415.9K | 75.6% |
| 408 | TTWO | TTWOUSDT (usdm-futures) | usdm-futures | 80.3% | 0.132 | SNOWUSDT | 100.0% | 0.1 | 200.7K | 42.6% |
| 409 | EWT | EWTUSDT (usdm-futures) | usdm-futures | 80.2% | 0.183 | BTCUSDT | 100.0% | 0.1 | 1.9M | 49.7% |
| 410 | ZRX | ZRXUSDT (spot) | spot, usdm-futures | 80.2% | 0.158 | BTCUSDT | 100.0% | 0.8 | 98.8K | 50.6% |
| 411 | KMNO | KMNOUSDT (spot) | spot, usdm-futures | 80.1% | 0.144 | BTCUSDT | 100.0% | 0.7 | 79.6K | 47.6% |
| 412 | PUMPBTC | PUMPBTCUSDT (usdm-futures) | usdm-futures | 80.1% | 0.192 | BTCUSDT | 100.0% | 0.1 | 365.3K | 85.6% |
| 413 | EDEN | EDENUSDT (spot) | spot, usdm-futures | 80.0% | 0.154 | BTCUSDT | 100.0% | 0.4 | 221.4K | 79.6% |
| 414 | SFP | SFPUSDT (spot) | spot, usdm-futures | 80.0% | 0.208 | BTCUSDT | 100.0% | 0.6 | 50.5K | 48.4% |
| 415 | CVC | CVCUSDT (spot) | spot, usdm-futures | 80.0% | 0.136 | UMAUSDT | 100.0% | 0.9 | 83.5K | 40.7% |
| 416 | DIS | DISUSDT (usdm-futures) | usdm-futures | 79.9% | 0.145 | SHAZUSDT | 100.0% | 0.1 | 218.7K | 36.3% |
| 417 | ORDER | ORDERUSDT (usdm-futures) | usdm-futures | 79.9% | 0.164 | BTCUSDT | 100.0% | 0.1 | 2.7M | 115.4% |
| 418 | OPG | OPGUSDT (spot) | spot, usdm-futures | 79.8% | 0.179 | BTCUSDT | 100.0% | 0.2 | 744.2K | 79.9% |
| 419 | QCOM | QCOMBUSDT (spot) | spot, usdm-futures | 79.8% | 0.122 | JPMUSDT | 100.0% | 0.1 | 111.2K | 92.9% |
| 420 | RARE | RAREUSDT (spot) | spot, usdm-futures | 79.7% | 0.123 | ZRXUSDT | 100.0% | 2.2 | 308.3K | 60.5% |
| 421 | RAVE | RAVEUSDT (usdm-futures) | usdm-futures | 79.6% | 0.205 | BTCUSDT | 100.0% | 0.1 | 9.1M | 123.2% |
| 422 | SPY | SPYBUSDT (spot) | spot, usdm-futures | 79.6% | 0.100 | AIAUSDT | 100.0% | 0.1 | 79.5K | 19.1% |
| 423 | DIA | DIAUSDT (spot) | spot, usdm-futures | 79.5% | 0.223 | BTCUSDT | 100.0% | 1 | 179.2K | 88.3% |
| 424 | ALCH | ALCHUSDT (usdm-futures) | usdm-futures | 79.4% | 0.238 | BTCUSDT | 100.0% | 0.1 | 1.3M | 104.6% |
| 425 | KAIA | KAIAUSDT (spot) | spot, usdm-futures | 79.3% | 0.161 | BMTUSDT | 100.0% | 0.9 | 261.5K | 72.7% |
| 426 | APE | APEUSDT (spot) | spot, usdm-futures | 79.2% | 0.215 | BTCUSDT | 100.0% | 0.3 | 731.7K | 74.5% |
| 427 | CHILLGUY | CHILLGUYUSDT (usdm-futures) | usdm-futures | 79.1% | 0.221 | BTCUSDT | 100.0% | 0 | 1.5M | 115.5% |
| 428 | PNUT | PNUTUSDT (spot) | spot, usdm-futures | 79.1% | 0.221 | BTCUSDT | 100.0% | 0.7 | 344.8K | 71.5% |
| 429 | BROCCOLI714 | BROCCOLI714USDT (spot) | spot, usdm-futures | 79.1% | 0.208 | BROCCOLIF3BUSDT | 100.0% | 0.4 | 493.1K | 86.6% |
| 430 | TOSHI | TOSHIUSDT (usdm-futures) | usdm-futures | 79.0% | 0.221 | BTCUSDT | 100.0% | 0.2 | 2.2M | 92.3% |
| 431 | ANIME | ANIMEUSDT (spot) | spot, usdm-futures | 78.9% | 0.133 | ASTRUSDT | 100.0% | 2.9 | 130.4K | 77.9% |
| 432 | PYTH | PYTHUSDT (spot) | spot, usdm-futures | 78.9% | 0.300 | BTCUSDT | 100.0% | 0.1 | 1.2M | 68.9% |
| 433 | GLM | GLMUSDT (spot) | spot, usdm-futures | 78.8% | 0.160 | FLOWUSDT | 100.0% | 0.8 | 74.2K | 48.1% |
| 434 | XBI | XBIUSDT (usdm-futures) | usdm-futures | 78.8% | 0.142 | HIMSUSDT | 100.0% | 0.1 | 262.9K | 35.8% |
| 435 | CSCO | CSCOUSDT (usdm-futures) | usdm-futures | 78.7% | 0.222 | CBRSBUSDT | 100.0% | 0.1 | 485.8K | 52.5% |
| 436 | BZ | BZUSDT (usdm-futures) | usdm-futures | 78.7% | 0.185 | XPTUSDT | 100.0% | 0.1 | 449.3M | 47.0% |
| 437 | ZK | ZKUSDT (spot) | spot, usdm-futures | 78.6% | 0.185 | BTCUSDT | 100.0% | 0.2 | 944.6K | 71.8% |
| 438 | BONK | BONKUSDT (spot) | spot, usdm-futures | 78.6% | 0.201 | BTCUSDT | 100.0% | 0.3 | 2.9M | 163.2% |
| 439 | BRETT | BRETTUSDT (usdm-futures) | usdm-futures | 78.5% | 0.266 | BTCUSDT | 100.0% | 0.1 | 3.6M | 120.4% |
| 440 | F | FUSDT (spot) | spot, usdm-futures | 78.5% | 0.229 | ATMUSDT | 100.0% | 1.3 | 273.9K | 104.6% |
| 441 | CFX | CFXUSDT (spot) | spot, usdm-futures | 78.4% | 0.233 | BTCUSDT | 100.0% | 0.3 | 744.1K | 61.2% |
| 442 | PUNDIX | PUNDIXUSDT (spot) | spot, usdm-futures | 78.3% | 0.164 | FLOWUSDT | 100.0% | 1 | 57.5K | 36.4% |
| 443 | DYDX | DYDXUSDT (spot) | spot, usdm-futures | 78.3% | 0.267 | BTCUSDT | 100.0% | 0.1 | 918K | 71.9% |
| 444 | SWARMS | SWARMSUSDT (usdm-futures) | usdm-futures | 78.1% | 0.249 | BTCUSDT | 100.0% | 0.1 | 1.1M | 80.3% |
| 445 | ARX | ARXUSDT (usdm-futures) | usdm-futures | 78.0% | 0.136 | SKYAIUSDT | 100.0% | 0.1 | 9.1M | 163.4% |
| 446 | QQQ | QQQBUSDT (spot) | spot, usdm-futures | 78.0% | 0.148 | VRTUSDT | 100.0% | 0 | 2.2M | 23.6% |
| 447 | COW | COWUSDT (spot) | spot, usdm-futures | 77.9% | 0.158 | BTCUSDT | 100.0% | 0.6 | 56.6K | 52.1% |
| 448 | PENG | PENGUSDT (usdm-futures) | usdm-futures | 77.9% | 0.244 | FLEXUSDT | 100.0% | 0.1 | 1.6M | 179.5% |
| 449 | STORJ | STORJUSDT (spot) | spot, usdm-futures | 77.8% | 0.153 | GLMUSDT | 100.0% | 1.3 | 65K | 42.1% |
| 450 | HYUNDAI | HYUNDAIUSDT (usdm-futures) | usdm-futures | 77.7% | 0.139 | QQQBUSDT | 100.0% | 0.1 | 1.9M | 72.9% |
| 451 | PIXEL | PIXELUSDT (spot) | spot, usdm-futures | 77.7% | 0.167 | BTCUSDT | 100.0% | 0.8 | 205K | 68.8% |
| 452 | RIVN | RIVNUSDT (usdm-futures) | usdm-futures | 77.5% | 0.185 | FLEXUSDT | 100.0% | 0.1 | 407.3K | 65.5% |
| 453 | ARK | ARKUSDT (spot) | spot, usdm-futures | 77.4% | 0.164 | DIAUSDT | 100.0% | 1.1 | 45.6K | 45.4% |
| 454 | MANA | MANAUSDT (spot) | spot, usdm-futures | 77.4% | 0.177 | BTCUSDT | 100.0% | 0.8 | 171.9K | 54.2% |
| 455 | MON | MONUSDT (usdm-futures) | usdm-futures | 77.2% | 0.257 | BTCUSDT | 100.0% | 0.1 | 12.1M | 110.1% |
| 456 | HAEDAL | HAEDALUSDT (spot) | spot, usdm-futures | 77.2% | 0.139 | PEOPLEUSDT | 100.0% | 0.3 | 219K | 85.7% |
| 457 | HOOD | HOODBUSDT (spot) | spot, usdm-futures | 77.2% | 0.132 | LLYUSDT | 100.0% | 0.1 | 122.8K | 89.4% |
| 458 | RSR | RSRUSDT (spot) | spot, usdm-futures | 77.1% | 0.217 | BTCUSDT | 100.0% | 0.4 | 419.1K | 74.7% |
| 459 | IOST | IOSTUSDT (spot) | spot, usdm-futures | 76.9% | 0.181 | BTCUSDT | 100.0% | 0.4 | 73.7K | 65.4% |
| 460 | ETHFI | ETHFIUSDT (spot) | spot, usdm-futures | 76.9% | 0.285 | BTCUSDT | 100.0% | 0.6 | 4.3M | 95.0% |
| 461 | REZ | REZUSDT (spot) | spot, usdm-futures | 76.9% | 0.157 | SFPUSDT | 100.0% | 0.3 | 422K | 76.2% |
| 462 | IOTA | IOTAUSDT (spot) | spot, usdm-futures | 76.8% | 0.129 | USUSDT | 100.0% | 0.4 | 471.8K | 106.2% |
| 463 | CKB | CKBUSDT (spot) | spot, usdm-futures | 76.7% | 0.214 | BTCUSDT | 100.0% | 0.5 | 136.3K | 52.9% |
| 464 | CFG | CFGUSDT (spot) | spot, usdm-futures | 76.6% | 0.161 | BTCUSDT | 100.0% | 0.4 | 1.3M | 95.6% |
| 465 | AVA | AVAUSDT (spot) | spot, usdm-futures | 76.6% | 0.165 | GLMUSDT | 100.0% | 0.7 | 90.2K | 52.5% |
| 466 | BABY | BABYUSDT (spot) | spot, usdm-futures | 76.6% | 0.202 | BTCUSDT | 100.0% | 0.7 | 298K | 73.2% |
| 467 | DOLO | DOLOUSDT (spot) | spot, usdm-futures | 76.5% | 0.149 | GRAMUSDT | 100.0% | 0.5 | 98.9K | 54.9% |
| 468 | MOCA | MOCAUSDT (usdm-futures) | usdm-futures | 76.4% | 0.166 | BTCUSDT | 100.0% | 0.3 | 358.6K | 75.7% |
| 469 | MORPHO | MORPHOUSDT (spot) | spot, usdm-futures | 76.3% | 0.258 | INITUSDT | 100.0% | 0.2 | 4M | 162.4% |
| 470 | ARKM | ARKMUSDT (spot) | spot, usdm-futures | 76.2% | 0.322 | BTCUSDT | 100.0% | 0.3 | 520.8K | 68.1% |
| 471 | NFLX | NFLXUSDT (usdm-futures) | usdm-futures | 76.2% | 0.120 | SNOWUSDT | 100.0% | 0.1 | 2.7M | 42.8% |
| 472 | MINA | MINAUSDT (spot) | spot, usdm-futures | 76.2% | 0.192 | COWUSDT | 100.0% | 1 | 165.8K | 62.7% |
| 473 | STX | STXUSDT (spot) | spot, usdm-futures | 76.1% | 0.238 | BTCUSDT | 100.0% | 0.4 | 299.5K | 51.2% |
| 474 | BERA | BERAUSDT (spot) | spot, usdm-futures | 76.0% | 0.198 | BTCUSDT | 100.0% | 0.5 | 1.2M | 114.8% |
| 475 | ETC | ETCUSDT (spot) | spot, usdm-futures, coinm-futures | 76.0% | 0.279 | BTCUSDT | 100.0% | 0.5 | 963.6K | 53.0% |
| 476 | CIEN | CIENUSDT (usdm-futures) | usdm-futures | 76.0% | 0.166 | XBIUSDT | 100.0% | 0.1 | 478.8K | 92.9% |
| 477 | NVDA | NVDABUSDT (spot) | spot, usdm-futures | 75.8% | 0.209 | QQQBUSDT | 100.0% | 0.1 | 889.8K | 46.2% |
| 478 | DUSK | DUSKUSDT (spot) | spot, usdm-futures | 75.7% | 0.157 | ARKUSDT | 100.0% | 0.4 | 159.5K | 87.9% |
| 479 | LDO | LDOUSDT (spot) | spot, usdm-futures | 75.6% | 0.285 | BTCUSDT | 100.0% | 0.1 | 3.7M | 79.3% |
| 480 | YFI | YFIUSDT (spot) | spot, usdm-futures | 75.6% | 0.197 | MANAUSDT | 100.0% | 0.4 | 268K | 52.7% |
| 481 | GRIFFAIN | GRIFFAINUSDT (usdm-futures) | usdm-futures | 75.5% | 0.236 | BTCUSDT | 100.0% | 0 | 1.1M | 91.7% |
| 482 | GIGGLE | GIGGLEUSDT (spot) | spot, usdm-futures | 75.4% | 0.191 | BTCUSDT | 100.0% | 0.1 | 1.1M | 70.7% |
| 483 | SONIC | SONICUSDT (usdm-futures) | usdm-futures | 75.4% | 0.274 | BTCUSDT | 100.0% | 0.1 | 361.6K | 73.7% |
| 484 | A | AUSDT (spot) | spot, usdm-futures | 75.3% | 0.174 | BTCUSDT | 100.0% | 0.5 | 180.1K | 72.5% |
| 485 | ORCA | ORCAUSDT (spot) | spot, usdm-futures | 75.2% | 0.168 | ASTRUSDT | 100.0% | 0.5 | 194.3K | 44.7% |
| 486 | DYM | DYMUSDT (spot) | spot, usdm-futures | 75.2% | 0.215 | BTCUSDT | 100.0% | 0.6 | 156.2K | 64.4% |
| 487 | TER | TERUSDT (usdm-futures) | usdm-futures | 75.0% | 0.211 | VRTUSDT | 100.0% | 0.1 | 930.3K | 103.0% |
| 488 | ETHW | ETHWUSDT (usdm-futures) | usdm-futures | 74.9% | 0.229 | BTCUSDT | 100.0% | 0.1 | 254.8K | 61.6% |
| 489 | URNM | URNMUSDT (usdm-futures) | usdm-futures | 74.9% | 0.206 | PENGUSDT | 100.0% | 0.1 | 484.1K | 45.6% |
| 490 | SIGN | SIGNUSDT (spot) | spot, usdm-futures | 74.9% | 0.171 | BTCUSDT | 100.0% | 0.6 | 120.9K | 57.2% |
| 491 | CVX | CVXUSDT (spot) | spot, usdm-futures | 74.8% | 0.265 | BTCUSDT | 100.0% | 0.4 | 556.6K | 74.5% |
| 492 | SLP | SLPUSDT (spot) | spot, usdm-futures | 74.6% | 0.166 | STXUSDT | 100.0% | 0.5 | 187K | 74.3% |
| 493 | EUL | EULUSDT (spot) | spot, usdm-futures | 74.6% | 0.205 | BTCUSDT | 100.0% | 0.5 | 202K | 73.1% |
| 494 | VELODROME | VELODROMEUSDT (spot) | spot, usdm-futures | 74.5% | 0.288 | BTCUSDT | 100.0% | 0.2 | 190.4K | 71.7% |
| 495 | MSFT | MSFTBUSDT (spot) | spot, usdm-futures | 74.5% | 0.157 | DISUSDT | 100.0% | 0.1 | 340.9K | 41.7% |
| 496 | MEME | MEMEUSDT (spot) | spot, usdm-futures | 74.3% | 0.189 | BTCUSDT | 100.0% | 0.5 | 305.9K | 77.7% |
| 497 | GMT | GMTUSDT (spot) | spot, usdm-futures | 74.2% | 0.186 | BTCUSDT | 100.0% | 0.4 | 408.7K | 74.9% |
| 498 | SPCX | SPCXBUSDT (spot) | spot, usdm-futures | 74.2% | 0.157 | QQQBUSDT | 100.0% | 0 | 21M | 72.9% |
| 499 | DEEP | DEEPUSDT (usdm-futures) | usdm-futures | 74.1% | 0.311 | BTCUSDT | 100.0% | 0.2 | 1.7M | 80.0% |
| 500 | AMZN | AMZNUSDT (usdm-futures) | usdm-futures | 74.0% | 0.199 | MSFTBUSDT | 100.0% | 0.1 | 13M | 41.8% |
| 501 | FLOCK | FLOCKUSDT (usdm-futures) | usdm-futures | 74.0% | 0.178 | BTCUSDT | 100.0% | 0.1 | 1.8M | 94.2% |
| 502 | LAYER | LAYERUSDT (spot) | spot, usdm-futures | 73.9% | 0.193 | PORTALUSDT | 100.0% | 0.7 | 361K | 60.8% |
| 503 | ZRO | ZROUSDT (spot) | spot, usdm-futures | 73.9% | 0.267 | BTCUSDT | 100.0% | 0.3 | 2.4M | 90.5% |
| 504 | CLANKER | CLANKERUSDT (usdm-futures) | usdm-futures | 73.7% | 0.252 | BTCUSDT | 100.0% | 0.2 | 663.1K | 81.7% |
| 505 | MEGA | MEGAUSDT (spot) | spot, usdm-futures | 73.6% | 0.293 | BTCUSDT | 100.0% | 0.1 | 1.2M | 72.5% |
| 506 | ID | IDUSDT (spot) | spot, usdm-futures | 73.6% | 0.152 | MEMEUSDT | 100.0% | 0.7 | 357.4K | 103.7% |

## Diagnostics

- Basis size selected: 506
- Coverage target reached: yes
- Mean pairwise absolute correlation: 0.033
- Maximum pairwise absolute correlation: 0.322
- Mean whole-market projection R²: 87.4%
- Median whole-market projection R²: 100.0%
- 10th-percentile whole-market projection R²: 50.0%
- Minimum whole-market projection R²: 46.0%

### Coverage by basis size

| Size | Mean R² | Median R² | 10th percentile R² | Minimum R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.8% | 1.4% | 0.0% | 0.0% |
| 5 | 5.6% | 1.7% | 0.3% | 0.0% |
| 10 | 6.7% | 2.2% | 0.6% | 0.2% |
| 15 | 7.7% | 2.6% | 1.0% | 0.4% |
| 20 | 8.8% | 3.2% | 1.4% | 0.8% |
| 25 | 10.0% | 3.6% | 1.8% | 1.1% |
| 30 | 11.1% | 4.2% | 2.2% | 1.6% |
| 35 | 12.2% | 4.7% | 2.6% | 1.9% |
| 40 | 13.4% | 5.2% | 3.0% | 2.3% |
| 45 | 14.5% | 5.8% | 3.4% | 2.6% |
| 50 | 15.5% | 6.3% | 3.8% | 2.9% |
| 55 | 16.7% | 6.9% | 4.3% | 3.3% |
| 60 | 17.7% | 7.4% | 4.6% | 3.7% |
| 65 | 18.8% | 7.9% | 5.1% | 4.1% |
| 70 | 19.9% | 8.5% | 5.5% | 4.6% |
| 75 | 21.0% | 9.0% | 6.0% | 5.0% |
| 80 | 22.0% | 9.5% | 6.4% | 5.3% |
| 85 | 23.0% | 9.9% | 6.9% | 5.7% |
| 90 | 24.1% | 10.4% | 7.3% | 6.0% |
| 95 | 25.2% | 11.1% | 7.8% | 6.7% |
| 100 | 26.2% | 11.6% | 8.1% | 7.0% |
| 105 | 27.2% | 12.0% | 8.6% | 7.4% |
| 110 | 28.1% | 12.6% | 9.0% | 7.7% |
| 115 | 29.2% | 13.2% | 9.4% | 8.2% |
| 120 | 30.2% | 13.9% | 9.9% | 8.6% |
| 125 | 31.2% | 14.5% | 10.3% | 9.1% |
| 130 | 32.2% | 15.2% | 10.7% | 9.6% |
| 135 | 33.2% | 15.7% | 11.2% | 9.9% |
| 140 | 34.1% | 16.2% | 11.6% | 10.3% |
| 145 | 35.0% | 16.8% | 12.1% | 10.8% |
| 150 | 35.9% | 17.3% | 12.5% | 11.1% |
| 155 | 36.9% | 17.9% | 12.9% | 11.6% |
| 160 | 37.9% | 18.5% | 13.4% | 12.2% |
| 165 | 38.8% | 19.2% | 13.8% | 12.6% |
| 170 | 39.8% | 20.0% | 14.4% | 13.0% |
| 175 | 40.7% | 20.5% | 14.9% | 13.4% |
| 180 | 41.6% | 21.1% | 15.3% | 13.9% |
| 185 | 42.6% | 21.8% | 15.7% | 14.3% |
| 190 | 43.4% | 22.3% | 16.1% | 14.7% |
| 195 | 44.4% | 23.2% | 16.6% | 15.1% |
| 200 | 45.3% | 23.8% | 17.2% | 15.7% |
| 205 | 46.2% | 24.5% | 17.6% | 16.0% |
| 210 | 47.1% | 25.2% | 18.0% | 16.5% |
| 215 | 48.0% | 25.9% | 18.5% | 17.0% |
| 220 | 48.8% | 26.7% | 19.1% | 17.5% |
| 225 | 49.7% | 27.4% | 19.5% | 18.0% |
| 230 | 50.5% | 28.5% | 20.0% | 18.5% |
| 235 | 51.4% | 29.1% | 20.5% | 19.1% |
| 240 | 52.2% | 29.9% | 21.0% | 19.4% |
| 245 | 53.0% | 30.6% | 21.5% | 20.0% |
| 250 | 53.8% | 31.0% | 21.9% | 20.5% |
| 255 | 54.7% | 32.0% | 22.3% | 20.9% |
| 260 | 55.6% | 33.1% | 22.8% | 21.3% |
| 265 | 56.3% | 33.8% | 23.3% | 21.8% |
| 270 | 57.1% | 34.6% | 23.8% | 22.5% |
| 275 | 57.9% | 35.6% | 24.3% | 22.9% |
| 280 | 58.6% | 36.5% | 24.8% | 23.3% |
| 285 | 59.6% | 38.3% | 25.3% | 23.8% |
| 290 | 60.4% | 39.5% | 25.9% | 24.2% |
| 295 | 61.2% | 41.1% | 26.3% | 24.6% |
| 300 | 62.0% | 42.2% | 26.9% | 25.0% |
| 305 | 62.7% | 44.1% | 27.4% | 25.6% |
| 310 | 63.5% | 45.9% | 27.8% | 26.0% |
| 315 | 64.2% | 48.0% | 28.3% | 26.6% |
| 320 | 64.9% | 49.3% | 28.7% | 26.9% |
| 325 | 65.6% | 51.3% | 29.3% | 27.4% |
| 330 | 66.3% | 54.1% | 29.7% | 27.8% |
| 335 | 67.0% | 57.7% | 30.4% | 28.4% |
| 340 | 67.8% | 63.1% | 30.9% | 29.0% |
| 345 | 68.5% | 73.8% | 31.5% | 29.3% |
| 350 | 69.2% | 91.8% | 31.9% | 29.8% |
| 355 | 69.8% | 100.0% | 32.3% | 30.4% |
| 360 | 70.5% | 100.0% | 32.8% | 30.8% |
| 365 | 71.1% | 100.0% | 33.2% | 31.4% |
| 370 | 71.8% | 100.0% | 33.7% | 31.7% |
| 375 | 72.5% | 100.0% | 34.5% | 32.4% |
| 380 | 73.1% | 100.0% | 35.0% | 33.0% |
| 385 | 73.8% | 100.0% | 35.5% | 33.4% |
| 390 | 74.4% | 100.0% | 36.0% | 33.9% |
| 395 | 75.0% | 100.0% | 36.4% | 34.4% |
| 400 | 75.6% | 100.0% | 37.0% | 34.9% |
| 405 | 76.3% | 100.0% | 37.5% | 35.3% |
| 410 | 76.9% | 100.0% | 38.2% | 35.9% |
| 415 | 77.5% | 100.0% | 38.6% | 36.1% |
| 420 | 78.1% | 100.0% | 39.3% | 36.7% |
| 425 | 78.7% | 100.0% | 39.9% | 37.3% |
| 430 | 79.3% | 100.0% | 40.4% | 37.7% |
| 435 | 79.8% | 100.0% | 41.0% | 38.1% |
| 440 | 80.5% | 100.0% | 41.4% | 38.5% |
| 445 | 81.0% | 100.0% | 42.0% | 39.2% |
| 450 | 81.6% | 100.0% | 42.7% | 39.7% |
| 455 | 82.2% | 100.0% | 43.1% | 40.5% |
| 460 | 82.7% | 100.0% | 43.7% | 40.9% |
| 465 | 83.2% | 100.0% | 44.4% | 41.3% |
| 470 | 83.8% | 100.0% | 45.2% | 41.9% |
| 475 | 84.3% | 100.0% | 45.8% | 42.3% |
| 480 | 84.8% | 100.0% | 46.6% | 43.0% |
| 485 | 85.3% | 100.0% | 47.2% | 43.4% |
| 490 | 85.8% | 100.0% | 47.6% | 44.1% |
| 495 | 86.4% | 100.0% | 48.3% | 44.8% |
| 500 | 86.9% | 100.0% | 49.3% | 45.3% |
| 505 | 87.3% | 100.0% | 49.8% | 45.8% |
| 506 | 87.4% | 100.0% | 50.0% | 46.0% |

### Selected-asset correlation matrix (first 24 basis assets)

| Asset | BTC | XNY | CYS | KGST | T | INIT | RIF | ANTHROPIC | B | IOTX | TOWNS | BAN | BEAT | O | BROCCOLIF3B | BOT | QUICK | GPS | PYR | NATGAS | HOT | JPM | STAR | ATM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 1.000 | 0.000 | 0.003 | -0.005 | 0.010 | 0.013 | -0.016 | -0.001 | -0.012 | 0.002 | 0.009 | 0.006 | 0.023 | 0.032 | 0.012 | -0.007 | 0.008 | 0.040 | 0.018 | -0.010 | 0.030 | -0.038 | 0.019 | 0.005 |
| XNY | 0.000 | 1.000 | -0.004 | -0.003 | -0.004 | -0.012 | -0.014 | -0.006 | -0.018 | -0.020 | -0.009 | 0.011 | -0.012 | 0.020 | 0.028 | -0.037 | 0.037 | -0.008 | 0.024 | 0.028 | 0.012 | -0.019 | 0.021 | -0.003 |
| CYS | 0.003 | -0.004 | 1.000 | -0.012 | 0.012 | 0.004 | -0.004 | -0.017 | 0.022 | 0.012 | 0.026 | -0.014 | 0.023 | -0.000 | -0.020 | 0.014 | 0.002 | -0.002 | 0.001 | -0.014 | 0.014 | -0.001 | 0.020 | 0.020 |
| KGST | -0.005 | -0.003 | -0.012 | 1.000 | 0.002 | -0.000 | -0.002 | -0.001 | -0.000 | -0.010 | 0.002 | 0.007 | 0.001 | -0.003 | -0.012 | 0.027 | 0.012 | 0.006 | 0.022 | 0.003 | -0.016 | 0.019 | 0.007 | -0.007 |
| T | 0.010 | -0.004 | 0.012 | 0.002 | 1.000 | 0.004 | -0.001 | 0.015 | 0.002 | -0.029 | -0.000 | 0.023 | 0.011 | -0.017 | 0.021 | 0.008 | -0.006 | 0.005 | -0.014 | 0.005 | 0.012 | -0.002 | 0.026 | -0.031 |
| INIT | 0.013 | -0.012 | 0.004 | -0.000 | 0.004 | 1.000 | -0.001 | -0.015 | 0.013 | 0.000 | 0.005 | -0.006 | 0.031 | -0.007 | -0.008 | -0.008 | 0.017 | -0.008 | -0.001 | 0.018 | 0.004 | 0.012 | -0.002 | -0.023 |
| RIF | -0.016 | -0.014 | -0.004 | -0.002 | -0.001 | -0.001 | 1.000 | -0.006 | -0.011 | 0.009 | -0.031 | 0.013 | -0.004 | 0.018 | -0.001 | -0.014 | -0.030 | -0.010 | 0.005 | -0.004 | 0.028 | -0.041 | 0.001 | -0.012 |
| ANTHROPIC | -0.001 | -0.006 | -0.017 | -0.001 | 0.015 | -0.015 | -0.006 | 1.000 | -0.002 | 0.003 | 0.011 | 0.026 | 0.017 | 0.013 | -0.005 | 0.023 | -0.023 | 0.023 | 0.025 | 0.025 | -0.005 | 0.000 | -0.018 | -0.010 |
| B | -0.012 | -0.018 | 0.022 | -0.000 | 0.002 | 0.013 | -0.011 | -0.002 | 1.000 | -0.013 | -0.008 | -0.016 | -0.002 | -0.011 | -0.013 | 0.015 | -0.002 | 0.005 | -0.019 | -0.028 | 0.010 | 0.012 | 0.028 | 0.040 |
| IOTX | 0.002 | -0.020 | 0.012 | -0.010 | -0.029 | 0.000 | 0.009 | 0.003 | -0.013 | 1.000 | 0.005 | 0.015 | -0.007 | 0.010 | -0.025 | -0.024 | 0.009 | 0.009 | 0.020 | 0.003 | 0.008 | -0.010 | 0.023 | 0.047 |
| TOWNS | 0.009 | -0.009 | 0.026 | 0.002 | -0.000 | 0.005 | -0.031 | 0.011 | -0.008 | 0.005 | 1.000 | 0.015 | 0.007 | -0.017 | 0.022 | 0.005 | -0.021 | 0.004 | -0.033 | 0.028 | -0.028 | -0.015 | -0.022 | 0.014 |
| BAN | 0.006 | 0.011 | -0.014 | 0.007 | 0.023 | -0.006 | 0.013 | 0.026 | -0.016 | 0.015 | 0.015 | 1.000 | 0.013 | 0.013 | 0.013 | 0.015 | -0.014 | -0.029 | -0.003 | 0.007 | 0.011 | -0.027 | -0.013 | -0.001 |
| BEAT | 0.023 | -0.012 | 0.023 | 0.001 | 0.011 | 0.031 | -0.004 | 0.017 | -0.002 | -0.007 | 0.007 | 0.013 | 1.000 | 0.007 | 0.003 | 0.003 | -0.003 | 0.025 | -0.005 | -0.018 | -0.019 | -0.002 | 0.028 | 0.012 |
| O | 0.032 | 0.020 | -0.000 | -0.003 | -0.017 | -0.007 | 0.018 | 0.013 | -0.011 | 0.010 | -0.017 | 0.013 | 0.007 | 1.000 | 0.007 | 0.009 | -0.019 | 0.010 | 0.020 | -0.000 | 0.014 | -0.010 | -0.003 | 0.022 |
| BROCCOLIF3B | 0.012 | 0.028 | -0.020 | -0.012 | 0.021 | -0.008 | -0.001 | -0.005 | -0.013 | -0.025 | 0.022 | 0.013 | 0.003 | 0.007 | 1.000 | -0.001 | -0.002 | -0.030 | 0.018 | 0.037 | -0.022 | -0.015 | -0.042 | 0.009 |
| BOT | -0.007 | -0.037 | 0.014 | 0.027 | 0.008 | -0.008 | -0.014 | 0.023 | 0.015 | -0.024 | 0.005 | 0.015 | 0.003 | 0.009 | -0.001 | 1.000 | 0.009 | 0.010 | -0.017 | 0.000 | -0.034 | 0.020 | -0.024 | 0.031 |
| QUICK | 0.008 | 0.037 | 0.002 | 0.012 | -0.006 | 0.017 | -0.030 | -0.023 | -0.002 | 0.009 | -0.021 | -0.014 | -0.003 | -0.019 | -0.002 | 0.009 | 1.000 | 0.014 | 0.017 | 0.006 | 0.019 | 0.018 | -0.006 | -0.032 |
| GPS | 0.040 | -0.008 | -0.002 | 0.006 | 0.005 | -0.008 | -0.010 | 0.023 | 0.005 | 0.009 | 0.004 | -0.029 | 0.025 | 0.010 | -0.030 | 0.010 | 0.014 | 1.000 | -0.007 | 0.011 | -0.024 | 0.003 | -0.006 | 0.008 |
| PYR | 0.018 | 0.024 | 0.001 | 0.022 | -0.014 | -0.001 | 0.005 | 0.025 | -0.019 | 0.020 | -0.033 | -0.003 | -0.005 | 0.020 | 0.018 | -0.017 | 0.017 | -0.007 | 1.000 | -0.005 | 0.001 | 0.018 | 0.021 | -0.006 |
| NATGAS | -0.010 | 0.028 | -0.014 | 0.003 | 0.005 | 0.018 | -0.004 | 0.025 | -0.028 | 0.003 | 0.028 | 0.007 | -0.018 | -0.000 | 0.037 | 0.000 | 0.006 | 0.011 | -0.005 | 1.000 | -0.038 | -0.043 | -0.018 | 0.012 |
| HOT | 0.030 | 0.012 | 0.014 | -0.016 | 0.012 | 0.004 | 0.028 | -0.005 | 0.010 | 0.008 | -0.028 | 0.011 | -0.019 | 0.014 | -0.022 | -0.034 | 0.019 | -0.024 | 0.001 | -0.038 | 1.000 | -0.002 | -0.002 | -0.026 |
| JPM | -0.038 | -0.019 | -0.001 | 0.019 | -0.002 | 0.012 | -0.041 | 0.000 | 0.012 | -0.010 | -0.015 | -0.027 | -0.002 | -0.010 | -0.015 | 0.020 | 0.018 | 0.003 | 0.018 | -0.043 | -0.002 | 1.000 | -0.006 | -0.008 |
| STAR | 0.019 | 0.021 | 0.020 | 0.007 | 0.026 | -0.002 | 0.001 | -0.018 | 0.028 | 0.023 | -0.022 | -0.013 | 0.028 | -0.003 | -0.042 | -0.024 | -0.006 | -0.006 | 0.021 | -0.018 | -0.002 | -0.006 | 1.000 | -0.004 |
| ATM | 0.005 | -0.003 | 0.020 | -0.007 | -0.031 | -0.023 | -0.012 | -0.010 | 0.040 | 0.047 | 0.014 | -0.001 | 0.012 | 0.022 | 0.009 | 0.031 | -0.032 | 0.008 | -0.006 | 0.012 | -0.026 | -0.008 | -0.004 | 1.000 |

The complete 506 × 506 matrix is retained in the JSON report.

### Least-covered eligible assets

A low R² here means the current basis does not explain much of that asset. Increase `--size` if these residuals matter.

| Asset | Symbol | Projection R² | Residual | Closest basis | Correlation |
| --- | --- | ---: | ---: | --- | ---: |
| EWJ | EWJUSDT | 46.0% | 73.5% | BTCUSDT | 0.184 |
| MOVR | MOVRUSDT | 46.0% | 73.5% | BTCUSDT | 0.172 |
| COMP | COMPUSDT | 46.1% | 73.4% | BTCUSDT | 0.241 |
| RAY | RAYUSDT | 46.2% | 73.4% | BTCUSDT | 0.297 |
| SUPER | SUPERUSDT | 46.2% | 73.3% | ARKUSDT | 0.187 |
| ARM | ARMBUSDT | 46.3% | 73.3% | TERUSDT | 0.188 |
| FLUX | FLUXUSDT | 46.3% | 73.3% | METISUSDT | 0.184 |
| THETA | THETAUSDT | 46.3% | 73.3% | BTCUSDT | 0.217 |
| SKY | SKYUSDT | 46.4% | 73.2% | BTCUSDT | 0.335 |
| IRYS | IRYSUSDT | 46.4% | 73.2% | BTCUSDT | 0.230 |
| USELESS | USELESSUSDT | 46.5% | 73.1% | BTCUSDT | 0.370 |
| CTK | CTKUSDT | 46.6% | 73.1% | GLMUSDT | 0.189 |
| ZM | ZMUSDT | 46.6% | 73.1% | SNOWUSDT | 0.257 |
| 我踏马来了 | 我踏马来了USDT | 46.7% | 73.0% | SCRUSDT | 0.135 |
| MAGIC | MAGICUSDT | 46.7% | 73.0% | ORCAUSDT | 0.217 |

## Interpretation

This is an empirical basis of historical return directions, not a portfolio allocation and not a profitability claim. Selection is sensitive to the window, listings, liquidity, and correlation regime. Validate stability across adjacent windows before using it for capital allocation.

