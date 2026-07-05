# Static Sigma And SMA Speed Benchmark

Date: 2026-07-03

Scope:

- BTCUSDT 1m historical candles.
- Ten 3-day UTC windows selected for trend, sideways bias, and churn/no-churn behavior.
- Both long and short sides enabled.
- `maxLeverage=1`, `longBorrowDepth=999`, `shortBorrowDepth=999`.
- Static sigma pairs: `buy/sell` = `0.1/0.1`, `0.1/0.3`, `0.3/0.1`, `0.3/0.3`.
- SMA profiles:
  - `default-sma`: `[1s, 60s, 600s, 1800s, 3600s, 4h, 12h]`
  - `all-sma-2x`: `[1s, 30s, 300s, 900s, 1800s, 2h, 6h]`

The `all-sma-2x` profile halves the primary source window, confirmation windows, sizing derivative window, candle-range windows, and the closest trend-sigma window if trend mode is used. In these static tests, trend sigma itself is off.

Runner:

```bash
npx tsx scripts/benchmark-static-sigma-speed-grid.ts --case-index 0
```

Omit `--case-index` to run all ten cases sequentially.

## Best Per Case

| case | market | best default SMA | return | best all-SMA-2x | return | delta |
|---|---:|---|---:|---|---:|---:|
| uptrend low churn | `+7.355%` | `0.3/0.1` | `+5.093%` | `0.3/0.1` | `+3.238%` | `-1.855%` |
| uptrend high churn | `+9.239%` | `0.3/0.1` | `+6.185%` | `0.3/0.1` | `-0.639%` | `-6.824%` |
| downtrend low churn | `-5.559%` | `0.1/0.3` | `+4.633%` | `0.1/0.3` | `+1.736%` | `-2.897%` |
| downtrend high churn | `-15.017%` | `0.1/0.3` | `+6.390%` | `0.1/0.3` | `+1.876%` | `-4.515%` |
| sideways high-bias high-churn | `+0.302%` | `0.1/0.1` | `-0.523%` | `0.1/0.1` | `-2.325%` | `-1.802%` |
| sideways high-bias low-churn | `-0.507%` | `0.1/0.3` | `+0.927%` | `0.1/0.3` | `-0.082%` | `-1.010%` |
| sideways low-bias high-churn | `-0.309%` | `0.1/0.1` | `+0.441%` | `0.1/0.1` | `-0.184%` | `-0.625%` |
| sideways low-bias low-churn | `-0.348%` | `0.3/0.1` | `+0.011%` | `0.3/0.1` | `+0.238%` | `+0.227%` |
| sideways mid-bias high-churn | `-0.064%` | `0.1/0.3` | `+1.330%` | `0.1/0.1` | `-0.432%` | `-1.762%` |
| sideways mid-bias low-churn | `+0.018%` | `0.1/0.1` | `-0.043%` | `0.1/0.1` | `-0.188%` | `-0.144%` |

If an oracle could pick the best static pair per case, default SMA averages `+2.444%` with 8 profitable cases. The all-SMA-2x version averages only `+0.324%` with 4 profitable cases.

## 3-State DP Oracle Capture

The oracle below is the current backtester `perfectMargin` benchmark with `maxLeverage=1`, after replacing the old friction-filtered path benchmark with a 3-state dynamic-programming oracle.

For each replayed price event, the oracle keeps the best state among:

```text
flat
long
short
```

It can hold, enter, exit, or switch sides at each event. Transitions pay one-way or two-way friction from fees and slippage. The linear oracle uses fixed starting notional. The compounded oracle reinvests current equity after each favorable move and transition.

Capture is:

```text
strategy return / 3-state DP oracle return
```

| case | DP return | DP PnL | compounded DP return | compounded DP PnL | best default-SMA capture | best all-SMA-2x capture |
|---|---:|---:|---:|---:|---:|---:|
| uptrend low churn | `+24.8650%` | `$2486.50` | `+27.9713%` | `$2797.13` | `20.4830%` | `13.0226%` |
| uptrend high churn | `+371.8128%` | `$37181.28` | `+3884.2163%` | `$388421.63` | `1.6634%` | `-0.1719%` |
| downtrend low churn | `+32.4971%` | `$3249.71` | `+38.0472%` | `$3804.72` | `14.2570%` | `5.3424%` |
| downtrend high churn | `+922.9067%` | `$92290.67` | `+941080.3062%` | `$94108030.62` | `0.6924%` | `0.2032%` |
| sideways high-bias high-churn | `+141.9180%` | `$14191.80` | `+308.2610%` | `$30826.10` | `-0.3682%` | `-1.6381%` |
| sideways high-bias low-churn | `+15.1373%` | `$1513.73` | `+16.2114%` | `$1621.14` | `6.1268%` | `-0.5427%` |
| sideways low-bias high-churn | `+97.0964%` | `$9709.64` | `+161.9472%` | `$16194.72` | `0.4542%` | `-0.1896%` |
| sideways low-bias low-churn | `+7.2904%` | `$729.04` | `+7.5021%` | `$750.21` | `0.1555%` | `3.2701%` |
| sideways mid-bias high-churn | `+127.4050%` | `$12740.50` | `+253.5243%` | `$25352.43` | `1.0439%` | `-0.3387%` |
| sideways mid-bias low-churn | `+9.8578%` | `$985.78` | `+10.2851%` | `$1028.51` | `-0.4381%` | `-1.9032%` |

This makes the weakness in high-churn windows more visible. The high-churn sideways cases have large theoretical path profit, but the current strategy captures little or negative value:

- high-bias high-churn sideways: best default-SMA capture `-0.3682%`
- low-bias high-churn sideways: best default-SMA capture `0.4542%`
- mid-bias high-churn sideways: best default-SMA capture `1.0439%`

That supports adding a dedicated bounded mean-reversion grid for sideways/high-churn regimes instead of trying to solve them only with sigma and SMA speed.

## Pair Averages

| SMA profile | buy/sell sigma | avg return | profitable cases | worst | best |
|---|---|---:|---:|---:|---:|
| default | `0.1/0.1` | `-0.101%` | `4/10` | `-1.175%` | `+0.851%` |
| default | `0.1/0.3` | `-1.066%` | `4/10` | `-11.893%` | `+6.390%` |
| default | `0.3/0.1` | `-1.233%` | `3/10` | `-12.811%` | `+6.185%` |
| default | `0.3/0.3` | `-8.436%` | `2/10` | `-29.461%` | `+3.547%` |
| all-SMA-2x | `0.1/0.1` | `-0.354%` | `2/10` | `-2.325%` | `+0.724%` |
| all-SMA-2x | `0.1/0.3` | `-2.773%` | `2/10` | `-11.557%` | `+1.876%` |
| all-SMA-2x | `0.3/0.1` | `-2.811%` | `2/10` | `-13.102%` | `+3.238%` |
| all-SMA-2x | `0.3/0.3` | `-16.929%` | `0/10` | `-35.686%` | `-2.577%` |

## SMA Trend Window Probe

I swept SMA windows from `5m` to `24h` on the same ten labeled 3-day cases. Labels were the selected case groups: two uptrend, two downtrend, six sideways.

The easy but mostly hindsight classifier is end-to-end SMA drift:

```text
if finalSma - initialSma > threshold => uptrend
if finalSma - initialSma < -threshold => downtrend
otherwise => sideways
```

Almost every tested window classified all ten cases with a threshold around `3%` drift over the 3-day window. This is useful for labeling a completed interval, but it is not a good live trading signal because it needs most of the interval to be known.

For a less hindsight-heavy signal, the strongest result was aggregate SMA slope persistence:

```text
slope = sma(now) - sma(oneWindowAgo)
trendVote = positiveSlopeShare - negativeSlopeShare
```

Then classify:

```text
trendVote > threshold  => uptrend
trendVote < -threshold => downtrend
otherwise              => sideways
```

Best windows on these cases:

| SMA window | slope lookback | vote threshold | classified |
|---:|---:|---:|---:|
| `3h` | `1x window` | `0.157` | `10/10` |
| `4h` | `1x window` | `0.191` | `10/10` |
| `6h` | `1x window` | `0.220` | `10/10` |
| `8h` | `1x window` | `0.218` | `10/10` |
| `6h` | `0.1x window` | `0.237` | `10/10` |
| `8h` | `0.1x window` | `0.239` | `10/10` |
| `12h` | `0.1x window` | `0.223` | `10/10` |

This does not mean the signal stayed correct for every minute inside the 3-day window. The sweep above classifies the completed interval by counting all slope signs in that interval. In a live rolling version, the signal can flip or go neutral during pullbacks and range swings.

The simpler median-slope classifier did not fully separate the regimes. Its best runs classified `9/10`; the miss was usually `sideways-mid-bias-high-churn`, which looked like an uptrend by median slope despite ending flat. That means a single latest or median SMA derivative is not enough for sideways/high-churn detection.

Practical fit:

- Use a `4h` or `6h` SMA as the trend window.
- Use slope persistence over a regime horizon, not just the latest derivative.
- Treat `abs(trendVote) < 0.20` as neutral/sideways.
- Treat `abs(trendVote) > 0.25` as directional enough to widen the favored sigma.
- The current 1h trend sigma derivative is too local: with the default `rateRatio=0.01`, it compares the 1h SMA to roughly 36 seconds earlier. That is responsive, but not a reliable trend classifier.

### Persistent Signal Search

I also checked more stateful candidates that should persist better than raw SMA slope:

- trend efficiency: net displacement divided by total path movement
- rolling regression `R^2` with slope sign
- moving-average stack alignment
- Donchian/channel breakout with trailing invalidation
- CUSUM/market-structure state
- range migration: current rolling range overlap versus previous rolling range

No single causal signal stayed correctly in the target state throughout all ten 3-day windows. The best candidates reduced flips, but they still misclassified sustained submoves inside sideways windows or stayed neutral through part of clean directional windows.

Best persistent tradeoffs:

| signal | parameters | avg correct | worst case | trend correct | sideways correct | avg flips |
|---|---|---:|---:|---:|---:|---:|
| Donchian state | `24h` channel, exit at lower/upper `25%`, min range `1%` | `64.0%` | `29.8%` | `62.5%` | `65.0%` | `3.0` |
| Displacement/range | `36h` displacement divided by `36h` range, threshold `0.65` | `64.0%` | `25.4%` | `57.2%` | `68.5%` | `3.6` |
| Range migration | compare two `12h` ranges, center drift `0.5%`, overlap `80%` | `63.8%` | `29.3%` | `46.5%` | `75.2%` | `4.5` |

The failure mode is consistent: some "sideways" windows contain long directional submoves, and some low-churn directional windows start with countertrend movement. A signal that remains correct for the whole completed 3-day label would need future information.

The useful design is therefore two separate persistent states:

- `rangeIdentity`: stable rolling range center and high range overlap; drives pure grid validity.
- `directionalThesis`: channel breakout or displacement/range state; drives anticipatory trend-entry grids.

Sideways grid invalidation should follow `rangeIdentity`, not the directional trend signal. Directional grids should follow `directionalThesis`, with hysteresis and explicit thesis invalidation.

## Full Return Grid

Pair notation is `buySigma/sellSigma`.

| case | profile | `0.1/0.1` | `0.1/0.3` | `0.3/0.1` | `0.3/0.3` |
|---|---|---:|---:|---:|---:|
| uptrend low churn | default | `-0.308%` | `-5.550%` | `+5.093%` | `+3.470%` |
| uptrend low churn | all-SMA-2x | `+0.441%` | `-6.827%` | `+3.238%` | `-6.605%` |
| uptrend high churn | default | `-1.175%` | `-11.893%` | `+6.185%` | `-28.502%` |
| uptrend high churn | all-SMA-2x | `-0.862%` | `-11.557%` | `-0.639%` | `-35.686%` |
| downtrend low churn | default | `+0.046%` | `+4.633%` | `-4.359%` | `+3.547%` |
| downtrend low churn | all-SMA-2x | `-0.433%` | `+1.736%` | `-5.594%` | `-2.577%` |
| downtrend high churn | default | `-0.321%` | `+6.390%` | `-12.811%` | `-29.461%` |
| downtrend high churn | all-SMA-2x | `+0.724%` | `+1.876%` | `-13.102%` | `-29.299%` |
| sideways high-bias high-churn | default | `-0.523%` | `-2.026%` | `-1.617%` | `-12.089%` |
| sideways high-bias high-churn | all-SMA-2x | `-2.325%` | `-4.983%` | `-4.619%` | `-26.385%` |
| sideways high-bias low-churn | default | `+0.067%` | `+0.927%` | `-0.745%` | `-0.427%` |
| sideways high-bias low-churn | all-SMA-2x | `-0.105%` | `-0.082%` | `-1.510%` | `-8.363%` |
| sideways low-bias high-churn | default | `+0.441%` | `-2.354%` | `-0.367%` | `-9.863%` |
| sideways low-bias high-churn | all-SMA-2x | `-0.184%` | `-4.223%` | `-1.790%` | `-26.238%` |
| sideways low-bias low-churn | default | `-0.043%` | `-1.331%` | `+0.011%` | `-1.232%` |
| sideways low-bias low-churn | all-SMA-2x | `-0.177%` | `-1.387%` | `+0.238%` | `-3.677%` |
| sideways mid-bias high-churn | default | `+0.851%` | `+1.330%` | `-2.939%` | `-7.818%` |
| sideways mid-bias high-churn | all-SMA-2x | `-0.432%` | `-1.606%` | `-3.716%` | `-26.128%` |
| sideways mid-bias low-churn | default | `-0.043%` | `-0.786%` | `-0.778%` | `-1.982%` |
| sideways mid-bias low-churn | all-SMA-2x | `-0.188%` | `-0.673%` | `-0.618%` | `-4.333%` |

## Interpretation

Static sigma behavior is strongly regime-dependent:

- Uptrend windows want `buySigma=0.3`, `sellSigma=0.1`.
- Downtrend windows want `buySigma=0.1`, `sellSigma=0.3`.
- Sideways/churn windows usually prefer `0.1/0.1`, or a one-sided bias only when the window has clear high/low path bias.
- `0.3/0.3` is consistently dangerous. It can realize large profits while leaving much larger underwater inventory. It averaged `-8.436%` on default SMA and `-16.929%` on all-SMA-2x.

The all-SMA-2x profile is not a general improvement. It catches more fast moves, but it also creates many more entries and much more leftover inventory. This matches the earlier signal-delay analysis: faster signals improve opportunity detection, but without inventory controls they mostly amplify churn.

## Dynamic Trend Fit Notes

I checked a few neutral-base exponential trend configs:

| trend config | avg return | profitable cases | worst | best |
|---|---:|---:|---:|---:|
| `a=0.1, b=0.02` | `-0.095%` | `4/10` | `-1.179%` | `+0.859%` |
| `a=0.1, b=0.05` | `-0.087%` | `4/10` | `-1.188%` | `+0.870%` |
| `a=0.1, b=0.1` | `-0.072%` | `4/10` | `-1.193%` | `+0.889%` |
| `a=0.1, b=0.2` | `-0.040%` | `5/10` | `-1.191%` | `+0.924%` |
| `a=0.12, b=0.1` | `-0.313%` | `4/10` | `-3.023%` | `+1.368%` |

These are stable but too weak. They behave close to static `0.1/0.1` and do not rotate into the strong directional winners. The previous `a=1` default is too wide and fails badly on high-churn windows.

A better dynamic fit should not be just a symmetric exponential sigma:

```text
buySigma  = a * e^( b * x)
sellSigma = a * e^(-b * x)
```

That shape has two problems:

- If `a` is high enough to reach directional `0.3` values, neutral/churn periods become too wide.
- If `a` is low enough for neutral/churn safety, the trend adjustment is too weak unless `b` is made aggressive, which risks unstable jumps.

Preferred direction:

```text
neutralSigma = 0.1
favoredSigma = 0.1 + 0.2 * trendStrength
unfavoredSigma = 0.1 or lower, plus side/inventory gate
```

Where:

- `trendStrength = clamp01(abs(smoothedTrendScore))`
- `trendDirection = sign(smoothedTrendScore)`
- `smoothedTrendScore` should combine 1h derivative with a higher-level confirmation, such as 3h/12h derivative or range position.
- In uptrend: target `buy/sell ~= 0.3/0.1`.
- In downtrend: target `buy/sell ~= 0.1/0.3`.
- In sideways high churn: target `0.1/0.1`, not faster SMAs.
- In sideways low churn: either stay very small or avoid trading because there is little movement to harvest.

Fast SMAs should be used as a probe layer, not as the only signal:

- fast source/confirmation can open small exploratory entries;
- slow confirmation should be required before scaling position;
- inventory pressure should block adding to a side that is already underwater;
- high churn should reduce max open inventory even if it increases signal count.

The current strategy can adapt to clean directional regimes through sigma choice, but it does not yet adapt safely to high-churn regimes. Sigma controls entry size, but without a regime gate and inventory-aware throttling it cannot distinguish "good frequent opportunities" from "fast reversals that build stale inventory".

## Sigmoid Sigma Fit Probe

Formula tested, interpreted per side:

```text
a = sigmoid(trend, slopeA)
b = sigmoid(-trend, slopeB)

sellSigma = sigmaX * a + (1 - a) * sigmaY
buySigma  = sigmaX * b + (1 - b) * sigmaY

sigmaX = 0.1
sigmaY = 0.3
```

This maps positive trend toward `buy/sell = 0.3/0.1`, and negative trend toward `0.1/0.3`. I fitted `slopeA` and `slopeB` against the best default-SMA static pair for each of the ten cases, using the current 1h trend-sigma signal.

Best fit:

```text
slopeA = 32.1
slopeB = 879.2
RMSE   = 0.08884 sigma points
```

Per-case fit:

| case | median 1h trend | target buy/sell | fitted buy/sell |
|---|---:|---:|---:|
| uptrend low churn | `+0.0084` | `0.3/0.1` | `0.2999/0.1867` |
| uptrend high churn | `+0.0204` | `0.3/0.1` | `0.3000/0.1684` |
| downtrend low churn | `-0.0038` | `0.1/0.3` | `0.1071/0.2060` |
| downtrend high churn | `-0.1076` | `0.1/0.3` | `0.1000/0.2939` |
| sideways high-bias high-churn | `-0.0064` | `0.1/0.1` | `0.1007/0.2102` |
| sideways high-bias low-churn | `-0.0025` | `0.1/0.3` | `0.1196/0.2040` |
| sideways low-bias high-churn | `0.0000` | `0.1/0.1` | `0.2004/0.2000` |
| sideways low-bias low-churn | `-0.0006` | `0.3/0.1` | `0.1747/0.2009` |
| sideways mid-bias high-churn | `+0.0152` | `0.1/0.3` | `0.3000/0.1760` |
| sideways mid-bias low-churn | `-0.0010` | `0.1/0.1` | `0.1579/0.2016` |

This formula cannot fit the current best static choices well enough as-is:

- Neutral trend always maps near `0.2/0.2`, but several sideways winners need `0.1/0.1`.
- Both sides cannot be low at the same time, because low buy requires negative trend while low sell requires positive trend.
- The current 1h trend median is close to zero in several directional or biased windows, so very large slopes are required; that makes the fit unstable.

A better sigmoid shape needs a separate neutral/churn gate:

```text
baseSigma = 0.1 when abs(trendStrength) is low or range/churn regime is active
favoredSigma = 0.1 + 0.2 * directionalConfidence
unfavoredSigma = 0.1
```

## Return-Optimized Sigmoid Sigma Probe

Date: 2026-07-04

I implemented the interpolation formula as `sigmaMode=sigmoid-trend`:

```text
a = sigmoid(trend, slopeA)
b = sigmoid(-trend, slopeB)

sellSigma = sigmaX * a + (1 - a) * sigmaY
buySigma  = sigmaX * b + (1 - b) * sigmaY
```

Test constants:

```text
sigmaX = 0.05
sigmaY = 0.3
```

`trend` is the existing relative SMA derivative scaled by the BTC reference price, with the trend SMA chosen by `trendSigmaWindowSec`. I ran a bounded return fit on the same ten 3-day windows, optimizing average return across all cases.

Runner:

```bash
npm run benchmark:sigmoid-sigma -- --windows-sec 43200 --slope-a 15 --slope-b 300
```

The full brute-force grid is too slow for routine use: one candidate across all ten cases takes about 60-70 seconds on this machine. The table below is therefore a bounded sweep/refinement, not an exhaustive global optimum.

| trend SMA | slopeA | slopeB | avg return | profitable cases | worst | best | avg capture |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1h` | `0` | `0` | `-3.2546%` | `3/10` | `-11.6331%` | `+1.8934%` | `-2.0445%` |
| `1h` | `25` | `400` | `-4.3612%` | `3/10` | `-15.0229%` | `+3.4450%` | `-2.6850%` |
| `1h` | `50` | `800` | `-4.6046%` | `3/10` | `-15.5085%` | `+3.5396%` | `-3.0111%` |
| `4h` | `25` | `400` | `-2.5864%` | `3/10` | `-13.8032%` | `+3.2125%` | `-2.1334%` |
| `4h` | `50` | `800` | `-2.5779%` | `3/10` | `-13.5451%` | `+3.1537%` | `-2.2983%` |
| `4h` | `100` | `1600` | `-2.6059%` | `3/10` | `-13.3706%` | `+3.1662%` | `-2.3035%` |
| `6h` | `50` | `800` | `-2.6321%` | `2/10` | `-10.0462%` | `+3.1273%` | `-3.4836%` |
| `12h` | `5` | `100` | `-1.7568%` | `3/10` | `-8.9100%` | `+3.0582%` | `-2.1515%` |
| `12h` | `10` | `200` | `-1.6153%` | `3/10` | `-7.7994%` | `+3.3019%` | `-2.4362%` |
| `12h` | `15` | `300` | `-1.6123%` | `3/10` | `-7.7964%` | `+3.3160%` | `-2.6083%` |
| `12h` | `25` | `400` | `-1.6632%` | `3/10` | `-8.0716%` | `+3.1827%` | `-2.8626%` |
| `12h` | `50` | `800` | `-1.9599%` | `3/10` | `-8.8763%` | `+2.9552%` | `-3.3569%` |
| `12h` | `100` | `1600` | `-2.1003%` | `2/10` | `-9.9147%` | `+3.0525%` | `-3.5341%` |

Best observed fit:

```text
trendSigmaWindowSec = 43200
slopeA = 15
slopeB = 300
sigmaX = 0.05
sigmaY = 0.3
avg return = -1.6123%
```

Per-case returns for that fit:

| case | return | capture |
|---|---:|---:|
| uptrend low churn | `+3.3160%` | `13.3362%` |
| uptrend high churn | `-7.7964%` | `-2.0969%` |
| downtrend low churn | `+1.7056%` | `5.2486%` |
| downtrend high churn | `+2.7695%` | `0.3001%` |
| sideways high-bias high-churn | `-4.5672%` | `-3.2182%` |
| sideways high-bias low-churn | `-0.4243%` | `-2.8031%` |
| sideways low-bias high-churn | `-4.8239%` | `-4.9681%` |
| sideways low-bias low-churn | `-0.9204%` | `-12.6245%` |
| sideways mid-bias high-churn | `-3.7758%` | `-2.9636%` |
| sideways mid-bias low-churn | `-1.6062%` | `-16.2939%` |

Result: this sigmoid interpolation is not competitive with even the static `0.1/0.1` default-SMA row (`-0.101%` average), and it is far below the per-case static oracle (`+2.444%` average). A slower `12h` trend input reduces the worst churn losses versus `1h`, but the formula still cannot express the most important sideways behavior: both sides low at the same time.

The useful takeaway is still consistent with the earlier fit notes: sigma interpolation needs an explicit neutral/range gate. Without that gate, neutral trend maps near the middle of `[0.05, 0.3]`, while the high-churn sideways cases need inventory throttling and near-`0.1/0.1` or lower behavior.
