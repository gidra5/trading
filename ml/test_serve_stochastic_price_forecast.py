from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from serve_stochastic_price_forecast import process_line, validate_request
from stochastic_price_forecast import StochasticPriceForecast, fit_training_basis, price_tree, two_tick_scenarios


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.request = {"id": "test", "observedPrice": 100.0, "features": [0.0] * 178}

    def test_requires_complete_finite_current_inputs(self):
        for field, value in (("features", [0.0] * 177), ("features", [float("inf")] * 178),
                             ("observedPrice", 0), ("observedPrice", True), ("id", float("inf"))):
            with self.subTest(field=field, value=str(value)[:40]):
                with self.assertRaises(ValueError):
                    validate_request({**self.request, field: value})
        with self.assertRaises(ValueError):
            validate_request({**self.request, "realizedNextPrice": 101})

    def test_bad_request_returns_explicit_serializable_error(self):
        for line in ("not json", '{"id": 1e999}', '{"id": "x", "observedPrice": NaN}',
                     json.dumps({**self.request, "features": []})):
            response = process_line(line, None, torch.device("cpu"))
            self.assertEqual(response["error"]["code"], "invalid-request")
            json.dumps(response, allow_nan=False)

    def test_worker_and_export_share_the_same_tree(self):
        rng = np.random.default_rng(44)
        targets = np.concatenate((np.zeros(100), rng.normal(0, 1e-4, 300))).astype(np.float32)
        basis = fit_training_basis(targets, 32)
        model = StochasticPriceForecast(torch.zeros(178), torch.ones(178), basis, (16, 16)).eval()
        features = torch.zeros(1, 178)
        arrays = two_tick_scenarios(model, features)
        expected = price_tree(100, *(array[0] for array in arrays))
        response = process_line(json.dumps(self.request), model, torch.device("cpu"))
        self.assertEqual(response, {"id": "test", "tree": expected})
        self.assertEqual(len(response["tree"]["children"]), 4)
        self.assertTrue(all(len(branch["children"]) == 4 for branch in response["tree"]["children"]))


if __name__ == "__main__":
    unittest.main()
