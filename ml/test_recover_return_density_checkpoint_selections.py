from __future__ import annotations

import unittest

from recover_return_density_checkpoint_selections import optimal_policy_events


def event(
    epoch: int,
    *,
    train_mse: float,
    validation_mse: float,
    train_correlation: float,
    validation_correlation: float,
    train_nll: float,
    validation_nll: float,
) -> dict:
    return {
        "epoch": epoch,
        "train": {"mse": train_mse, "correlation": train_correlation},
        "validation": {
            "mse": validation_mse,
            "correlation": validation_correlation,
        },
        "trainDistribution": {"negativeLogLikelihood": train_nll},
        "validationDistribution": {"negativeLogLikelihood": validation_nll},
    }


class DensityCheckpointSelectionTest(unittest.TestCase):
    def test_each_policy_selects_its_own_optimum(self) -> None:
        events = [
            event(
                0, train_mse=3, validation_mse=1,
                train_correlation=1, validation_correlation=3,
                train_nll=3, validation_nll=2,
            ),
            event(
                1, train_mse=1, validation_mse=3,
                train_correlation=3, validation_correlation=1,
                train_nll=2, validation_nll=3,
            ),
            event(
                2, train_mse=2, validation_mse=2,
                train_correlation=2, validation_correlation=2,
                train_nll=1, validation_nll=1,
            ),
        ]
        selected = optimal_policy_events(events)
        self.assertEqual(selected["train-mse"]["epoch"], 1)
        self.assertEqual(selected["validation-mse"]["epoch"], 0)
        self.assertEqual(selected["train-correlation"]["epoch"], 1)
        self.assertEqual(selected["validation-correlation"]["epoch"], 0)
        self.assertEqual(selected["train-nll"]["epoch"], 2)
        self.assertEqual(selected["validation-nll"]["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
