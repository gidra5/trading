# BTCUSDT one-second zero-change gaps

## Definition

A zero-change gap is a maximal positive run of adjacent one-second close-to-close
returns equal to zero. Only runs bounded by nonzero returns inside the selected
window are counted; the at-most-two boundary-censored runs are excluded.

Source: complete BTCUSDT spot one-second shards from 2021-07-25 through the
exclusive common endpoint 2026-07-25.

## Results

| Window | Zero returns | Completed gaps | Mean | p50 | p90 | p95 | p99 | p99.9 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 5 years | 35.73% | 28,195,814 | 2.00s | 1s | 4s | 5s | 8s | 14s | 16,202s |
| Trailing 365d | 46.21% | 6,896,110 | 2.11s | 1s | 4s | 5s | 9s | 15s | 67s |
| Trailing 90d | 48.78% | 1,727,163 | 2.20s | 2s | 4s | 6s | 9s | 16s | 52s |
| Trailing 30d | 47.42% | 586,407 | 2.10s | 1s | 4s | 5s | 9s | 15s | 48s |
| Trailing 7d | 48.83% | 141,557 | 2.09s | 1s | 4s | 5s | 9s | 14s | 48s |

The ordinary distribution is concentrated at very short gaps: the full-history
median is 1 second, 99% finish within 8 seconds, and only 0.6465% last at least
10 seconds. Recent windows have a larger zero-return mass and slightly longer
typical gaps, but a similar short-run shape.

## Exceptional gaps

The full-history maximum is not representative of ordinary price inactivity.
Three isolated multi-hour flat runs dominate the extreme support:

| Duration | Start (UTC) | End (UTC) |
| ---: | --- | --- |
| 16,202s (4h 30m 2s) | 2021-08-13 01:59:59 | 2021-08-13 06:30:01 |
| 9,156s (2h 32m 36s) | 2023-03-24 11:27:24 | 2023-03-24 14:00:00 |
| 7,201s (2h 0m 1s) | 2021-09-29 07:00:00 | 2021-09-29 09:00:01 |

These are exchange-outage or maintenance-like episodes in the candle history.
After them, the next-longest completed gap is 81 seconds. They should be treated
as a separate data-quality/market-availability regime when modeling normal
microstructure.

The exact machine-readable probability masses and tail statistics are generated
at `data/benchmarks/zero-change-gaps.json` by
`scripts/analyze-zero-change-gaps.ts`.

## Parametric fit

Let $L\in\{1,2,\ldots\}$ be a completed zero-change gap length in seconds. The
ordinary full-history distribution, excluding the three multi-hour
availability events, is fitted by a two-component beta-geometric mixture. Its
survival function is

$$
\Pr(L\ge k)=
0.655505\frac{B(10.72309,11.60739+k-1)}{B(10.72309,11.60739)}
+0.344495\frac{B(5.42687,2.76766+k-1)}{B(5.42687,2.76766)},
$$

and its exact probability mass is

$$
\Pr(L=k)=
0.655505\frac{B(11.72309,11.60739+k-1)}{B(10.72309,11.60739)}
+0.344495\frac{B(6.42687,2.76766+k-1)}{B(5.42687,2.76766)}.
$$

Here $B(a,b)$ is the beta function. Equivalently, each component draws a
per-second termination probability from a beta distribution and then draws the
gap length from a geometric distribution. The asymptotic ordinary tail is

$$
\Pr(L\ge k)\propto k^{-5.42687},\qquad
\Pr(L=k)\propto k^{-6.42687}.
$$

This is an asymptotic tail, not a finite cutoff; the multi-hour events remain a
separate exchange-availability regime.
