# Greedy information basis for the next one-second return

Generated 2026-08-16T06:04:14.540Z. The exact quartile interaction search uses 39,438,595 held-out-sampled targets from the same five-year BTCUSDT one-second history.

## Result

The selected 5-indicator basis adds 0.15747885 bits/target beyond the latest-return state. Together with that baseline, the cumulative held-out gain over the unconditional distribution is 0.73401942 bits/target.

| order | indicator | family | individual quartile gain | marginal gain when added | cumulative basis gain | retained individual information | positive years |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | RSI(2s) | RSI | 0.071437005 | 0.071437005 | 0.071437005 | 100.00% | 4/4 |
| 2 | EMA acceleration(n=2s,k=1s) | EMA acceleration | 0.055712850 | 0.046348059 | 0.11778506 | 83.19% | 4/4 |
| 3 | EMA slope(n=8s,k=8s) | EMA slope | 0.032483896 | 0.024508952 | 0.14229402 | 75.45% | 4/4 |
| 4 | EMA acceleration(n=2s,k=2s) | EMA acceleration | 0.025725255 | 0.010725766 | 0.15301978 | 41.69% | 4/4 |
| 5 | Price−EMA(8s) | EMA | 0.037623316 | 0.0044590661 | 0.15747885 | 11.85% | 4/4 |

The retained-information column is `conditional marginal gain / standalone indicator gain`. Values far below 100% reveal redundancy with features already in the basis; values above 100% indicate complementary interaction.

## Practical stopping points

- The first three indicators capture 90.36% of the maximum tested five-feature information.
- The first four capture 97.17%.
- The full basis improves geometric assigned probability by 11.534% beyond the latest-return state; latest return plus basis improves it by 66.327% over the unconditional distribution.
- Use the first three as the compact basis, the first four when a small extra interaction cost is acceptable, and all five only when maximizing held-out distributional score matters more than minimality.

## Selection alternatives

### Step 1: after history only (166 candidates)

| candidate | family | marginal bits | cumulative bits | positive years |
|---|---|---:|---:|---:|
| RSI(2s) | RSI | 0.071437005 | 0.071437005 | 4/4 |
| EMA slope(n=2s,k=2s) | EMA slope | 0.060616340 | 0.060616340 | 4/4 |
| EMA acceleration(n=2s,k=1s) | EMA acceleration | 0.055712850 | 0.055712850 | 4/4 |
| RSI(4s) | RSI | 0.051265926 | 0.051265926 | 4/4 |
| EMA slope(n=2s,k=4s) | EMA slope | 0.049828078 | 0.049828078 | 4/4 |
| EMA slope(n=4s,k=2s) | EMA slope | 0.049018762 | 0.049018762 | 4/4 |
| Price−EMA(2s) | EMA | 0.047155220 | 0.047155220 | 4/4 |
| EMA slope(n=2s,k=1s) | EMA slope | 0.047155220 | 0.047155220 | 4/4 |
| EMA slope(n=4s,k=4s) | EMA slope | 0.042645794 | 0.042645794 | 4/4 |
| Price−EMA(4s) | EMA | 0.041943942 | 0.041943942 | 4/4 |
| EMA slope(n=4s,k=1s) | EMA slope | 0.041943942 | 0.041943942 | 4/4 |
| EMA slope(n=2s,k=8s) | EMA slope | 0.041154341 | 0.041154341 | 4/4 |

### Step 2: after rsi-2 (165 candidates)

| candidate | family | marginal bits | cumulative bits | positive years |
|---|---|---:|---:|---:|
| EMA acceleration(n=2s,k=1s) | EMA acceleration | 0.046348059 | 0.11778506 | 4/4 |
| EMA acceleration(n=4s,k=2s) | EMA acceleration | 0.039580442 | 0.11101745 | 4/4 |
| EMA slope(n=8s,k=8s) | EMA slope | 0.037252344 | 0.10868935 | 4/4 |
| EMA acceleration(n=8192s,k=8s) | EMA acceleration | 0.036908684 | 0.10834569 | 4/4 |
| EMA acceleration(n=8s,k=2s) | EMA acceleration | 0.036868436 | 0.10830544 | 4/4 |
| EMA slope(n=2s,k=2s) | EMA slope | 0.036862740 | 0.10829975 | 4/4 |
| EMA acceleration(n=4096s,k=8s) | EMA acceleration | 0.036589121 | 0.10802613 | 4/4 |
| EMA acceleration(n=4s,k=4s) | EMA acceleration | 0.036545981 | 0.10798299 | 4/4 |
| EMA acceleration(n=2s,k=4s) | EMA acceleration | 0.036374515 | 0.10781152 | 4/4 |
| EMA acceleration(n=2048s,k=8s) | EMA acceleration | 0.036184065 | 0.10762107 | 4/4 |
| MACD line(6,13,5) | MACD | 0.035709890 | 0.10714689 | 4/4 |
| EMA slope(n=8s,k=4s) | EMA slope | 0.035535619 | 0.10697262 | 4/4 |

### Step 3: after rsi-2 + ema-acceleration-2-1 (164 candidates)

| candidate | family | marginal bits | cumulative bits | positive years |
|---|---|---:|---:|---:|
| EMA slope(n=8s,k=8s) | EMA slope | 0.024508952 | 0.14229402 | 4/4 |
| EMA acceleration(n=8192s,k=8s) | EMA acceleration | 0.024328661 | 0.14211372 | 4/4 |
| EMA acceleration(n=4096s,k=8s) | EMA acceleration | 0.023983410 | 0.14176847 | 4/4 |
| EMA acceleration(n=2048s,k=8s) | EMA acceleration | 0.023592128 | 0.14137719 | 4/4 |
| EMA slope(n=4s,k=8s) | EMA slope | 0.023000215 | 0.14078528 | 4/4 |
| EMA acceleration(n=512s,k=8s) | EMA acceleration | 0.022727217 | 0.14051228 | 4/4 |
| MACD line(6,13,5) | MACD | 0.022632762 | 0.14041783 | 4/4 |
| MACD line(3,7,3) | MACD | 0.022314267 | 0.14009933 | 4/4 |
| EMA slope(n=8s,k=4s) | EMA slope | 0.022133336 | 0.13991840 | 4/4 |
| EMA acceleration(n=128s,k=8s) | EMA acceleration | 0.021714685 | 0.13949975 | 4/4 |
| EMA slope(n=8s,k=16s) | EMA slope | 0.021329539 | 0.13911460 | 4/4 |
| EMA acceleration(n=4s,k=2s) | EMA acceleration | 0.021258481 | 0.13904354 | 4/4 |

### Step 4: after rsi-2 + ema-acceleration-2-1 + ema-slope-8-8 (32 candidates)

| candidate | family | marginal bits | cumulative bits | positive years |
|---|---|---:|---:|---:|
| EMA acceleration(n=2s,k=2s) | EMA acceleration | 0.010725766 | 0.15301978 | 4/4 |
| EMA acceleration(n=4s,k=2s) | EMA acceleration | 0.010430270 | 0.15272429 | 4/4 |
| EMA acceleration(n=8s,k=4s) | EMA acceleration | 0.010356885 | 0.15265090 | 4/4 |
| EMA slope(n=4s,k=4s) | EMA slope | 0.010036558 | 0.15233057 | 4/4 |
| EMA acceleration(n=4096s,k=4s) | EMA acceleration | 0.0098546025 | 0.15214862 | 4/4 |
| EMA acceleration(n=8192s,k=4s) | EMA acceleration | 0.0098106377 | 0.15210465 | 4/4 |
| EMA acceleration(n=2048s,k=4s) | EMA acceleration | 0.0097710295 | 0.15206505 | 4/4 |
| EMA acceleration(n=512s,k=4s) | EMA acceleration | 0.0096084797 | 0.15190250 | 4/4 |
| MACD line(3,7,3) | MACD | 0.0088126006 | 0.15110662 | 4/4 |
| Price−EMA(8s) | EMA | 0.0085328145 | 0.15082683 | 4/4 |
| EMA slope(n=8s,k=1s) | EMA slope | 0.0085328145 | 0.15082683 | 4/4 |
| EMA acceleration(n=4s,k=4s) | EMA acceleration | 0.0084432427 | 0.15073726 | 4/4 |

### Step 5: after rsi-2 + ema-acceleration-2-1 + ema-slope-8-8 + ema-acceleration-2-2 (12 candidates)

| candidate | family | marginal bits | cumulative bits | positive years |
|---|---|---:|---:|---:|
| Price−EMA(8s) | EMA | 0.0044590661 | 0.15747885 | 4/4 |
| EMA slope(n=8s,k=1s) | EMA slope | 0.0044590661 | 0.15747885 | 4/4 |
| MACD line(3,7,3) | MACD | 0.0032568921 | 0.15627667 | 3/4 |
| EMA slope(n=4s,k=4s) | EMA slope | 0.0023271192 | 0.15534690 | 3/4 |
| EMA acceleration(n=8s,k=4s) | EMA acceleration | 0.0019913714 | 0.15501115 | 2/4 |
| EMA acceleration(n=4096s,k=4s) | EMA acceleration | 0.0015953105 | 0.15461509 | 3/4 |
| EMA acceleration(n=8192s,k=4s) | EMA acceleration | 0.0015857066 | 0.15460549 | 3/4 |
| EMA acceleration(n=2048s,k=4s) | EMA acceleration | 0.0014829668 | 0.15450275 | 3/4 |
| EMA acceleration(n=512s,k=4s) | EMA acceleration | 0.0011362469 | 0.15415603 | 3/4 |
| EMA acceleration(n=4s,k=4s) | EMA acceleration | 0.0010927984 | 0.15411258 | 3/4 |
| EMA slope(n=8s,k=16s) | EMA slope | 0.00051885760 | 0.15353864 | 2/4 |
| EMA acceleration(n=4s,k=2s) | EMA acceleration | -0.00028109034 | 0.15273869 | 2/4 |

## Objective and method

For history state `H`, current basis `B`, candidate `F`, and next-return cell `R`, the selected feature maximizes:

```text
mean_test[log2 P_train(R | H, B, F) - log2 P_train(R | H, B)]
```

Each indicator is reduced to 4 equal-mass year-0 cells. Interactions are counted exactly; annual test windows use only earlier years for their probability tables. A candidate must be positive in all four years when such candidates exist.

## Limitations

- This is greedy forward selection, not an exhaustive search over all indicator subsets.
- Quartile cells make exact interactions statistically and computationally tractable; a continuous model can retain more within-cell information.
- The search is capped at 5 indicators because exact state count grows as 4^d. Steps 1–3 scan all indicators; step 4 carries the strongest 32 prior conditional candidates and step 5 carries the strongest 12.
- Every indicator is a deterministic transform of price history, so this is a compact predictive representation rather than new market information beyond the raw history.
- The objective is distributional log score, not trading PnL after spread, fees, latency, and market impact.

## Reproducibility

```text
node --conditions=development --import tsx scripts/analyze-indicator-information-basis.ts
```

Complete rankings are stored in `data/benchmarks/indicator-information-basis.json`.
