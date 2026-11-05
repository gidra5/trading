"""Persistent causal JSONL inference worker for the Oracle CRRA bot provider.

Input: {"id": ..., "observedPrice": positive_number, "features": [178 numbers]}.
Output: {"id": ..., "tree": {"children": [...]}} or {"id": ..., "error": {...}}.
Only protocol responses go to stdout. No future prices/features are accepted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from stochastic_price_forecast import load_forecast, price_tree, two_tick_scenarios


def validate_request(request: object) -> tuple[object, float, np.ndarray]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if "id" not in request:
        raise ValueError("request requires id")
    if isinstance(request["id"], bool) or not isinstance(request["id"], (str, int)):
        raise ValueError("id must be a string or integer")
    if set(request) - {"id", "observedPrice", "features"}:
        raise ValueError("request contains unsupported fields; provide current observed inputs only")
    price = request.get("observedPrice")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError("observedPrice must be a finite positive number")
    features = request.get("features")
    if not isinstance(features, list) or len(features) != 178:
        raise ValueError("features must contain exactly 178 numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in features):
        raise ValueError("features must contain only finite numbers")
    values = np.asarray(features, dtype=np.float64)
    if np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError("features exceed the model's finite float32 input domain")
    return request["id"], float(price), values.astype(np.float32)


def reject_nonfinite_json(value: str):
    raise ValueError("non-finite JSON constant: " + value)


def process_line(line: str, model, device: torch.device) -> dict:
    request_id = None
    try:
        request = json.loads(line, parse_constant=reject_nonfinite_json)
        if isinstance(request, dict):
            candidate_id = request.get("id")
            if isinstance(candidate_id, (str, int)) and not isinstance(candidate_id, bool):
                request_id = candidate_id
        request_id, observed_price, values = validate_request(request)
    except (ValueError, TypeError, OverflowError) as error:
        return {"id": request_id, "error": {"code": "invalid-request", "message": str(error)}}
    try:
        features = torch.from_numpy(values[None, :]).to(device)
        first_returns, first_probabilities, second_returns, second_probabilities = two_tick_scenarios(model, features, 3)
        tree = price_tree(observed_price, first_returns[0], first_probabilities[0], second_returns[0], second_probabilities[0])
        # A failed numerical forecast is an explicit error, never silently replaced.
        response = {"id": request_id, "tree": tree}
        json.dumps(response, allow_nan=False)
        return response
    except Exception as error:
        return {"id": request_id, "error": {"code": "inference-failed", "message": str(error)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/training/runs/stochastic-crra-production178-zeroatom-k256-v1/best.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    selected_device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(selected_device)
    model, _payload = load_forecast(args.checkpoint, device)
    for line in sys.stdin:
        if not line.strip():
            continue
        response = process_line(line, model, device)
        sys.stdout.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
