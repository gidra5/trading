"""Independent synthetic fit-head diagnostic; never loads market data or models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_sign_size_forecast import fit_head
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "data/training/runs/stochastic-crra-sign-size-v1/synthetic-learnability-check.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(74293)
    train = rng.normal(size=(12000, 178)).astype(np.float32)
    validation = rng.normal(size=(4000, 178)).astype(np.float32)

    def targets(x):
        probability = 1 / (1 + np.exp(-4*x[:, 0] + x[:, 1]))
        sign = rng.random(len(x)) < probability
        up = np.exp(.7*x[:, 0] + .35*x[:, 1] + rng.normal(0, .15, len(x)))
        down = np.exp(-.55*x[:, 0] + .45*x[:, 2] + rng.normal(0, .15, len(x)))
        return sign, up, down

    train_targets, validation_targets = targets(train), targets(validation)
    report = {"seed": 74293, "trainExamples": len(train), "validationExamples": len(validation),
              "features": 178, "independentGeneratedSplits": True, "realMarketDataLoaded": False,
              "parameters": "unchanged production fit_head defaults", "heads": {}}
    for kind, y, yv in zip(("sign", "up", "down"), train_targets, validation_targets):
        if kind == "sign":
            constant = float(y.mean())
        else:
            scale = float(y.mean())
            y, yv = y / scale, yv / scale
            constant = 1.
        model, fitting = fit_head(kind, train, y, validation, yv)
        assert model is not None and fitting["bestIteration"] > 0, (kind, fitting)
        if kind == "sign":
            prediction = model.predict_proba(validation)[:, 1]
            loss = lambda p: float(-np.mean(yv*np.log(p)+(1-yv)*np.log1p(-p)))
            baseline, selected = loss(constant), loss(prediction)
        else:
            prediction = model.predict(validation)
            assert np.isfinite(prediction).all() and (prediction > 0).all()
            baseline, selected = float(np.mean((yv-constant)**2)), float(np.mean((yv-prediction)**2))
        assert selected < baseline * .8, (kind, selected, baseline)
        report["heads"][kind] = {**fitting, "metric": "logLoss" if kind == "sign" else "normalizedMSE",
                                  "constantMetric": baseline, "fittedMetric": selected,
                                  "relativeReduction": 1-selected/baseline}
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
