#!/usr/bin/env python3
"""Bin BTCUSDT aggTrade JSONL.GZ into causal buy/sell count arrays.

Preserves input order; for pure binning, equal-ms order does not change counts.
For event-level modeling, consume the raw file directly and preserve record order.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np


def build_counts(path: str, bin_ms: int = 1000):
    first = last = None
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            o = json.loads(line)
            t = int(o["timestampMs"])
            if first is None:
                first = t
            last = t
            side = str(o["aggressorSide"]).lower()
            j = 0 if side.startswith("b") else 1
            rows.append((t, j))
    if first is None:
        raise ValueError("empty input")
    t0 = (first // bin_ms) * bin_ms
    n = (last - t0) // bin_ms + 1
    y = np.zeros((n, 2), dtype=np.int32)
    for t, j in rows:
        y[(t - t0) // bin_ms, j] += 1
    return y, t0, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--bin-ms", type=int, default=1000)
    args = ap.parse_args()
    y, t0, last = build_counts(args.input, args.bin_ms)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, Y=y, t0=np.int64(t0), last=np.int64(last), bin_ms=np.int64(args.bin_ms))
    print("shape", y.shape, "means", y.mean(0), "total", y.sum(), "t0", t0, "last", last)

if __name__ == "__main__":
    main()
