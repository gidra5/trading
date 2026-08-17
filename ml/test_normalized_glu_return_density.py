from __future__ import annotations

import unittest

import numpy as np
import torch

from normalized_glu_return_density import NormalizedGluReturnDensity
from return_knot_density import KnotDensityContract, ReturnTransform


class NormalizedGluReturnDensityTest(unittest.TestCase):
    def test_direct_path_head_has_one_density_per_lead(self) -> None:
        density = KnotDensityContract(
            transform=ReturnTransform(
                alpha=4.0, location_bps=0.0, scale_bps=2.3
            ),
            knots_unit=np.asarray([0.0, 0.25, 0.75, 1.0]),
            prior_component_masses=np.asarray([0.1, 0.4, 0.4, 0.1]),
            source_file="test",
            source_fit="test",
        )
        model = NormalizedGluReturnDensity(
            torch.zeros(120),
            torch.ones(120),
            density,
            return_count=4,
            widths=(8,),
            dropout=0,
            dropout_rate=0,
        )
        logits = model.raw_density_logits(torch.zeros((3, 120)))
        self.assertEqual(logits.shape, (3, 4, 4))
        log_masses = model.log_component_masses(torch.zeros((3, 120)))
        expectation, _mode = model.point_predictions_from_log_masses(
            log_masses, include_mode=False
        )
        self.assertEqual(expectation.shape, (3, 4))
        torch.testing.assert_close(
            log_masses.exp(),
            model.density_prior_masses[None, None, :].expand(3, 4, -1),
        )

    def test_joint_logits_condition_on_realized_prefix(self) -> None:
        density = KnotDensityContract(
            transform=ReturnTransform(
                alpha=4.0, location_bps=0.0, scale_bps=2.3
            ),
            knots_unit=np.asarray([0.0, 0.25, 0.75, 1.0]),
            prior_component_masses=np.asarray([0.1, 0.4, 0.4, 0.1]),
            source_file="test",
            source_fit="test",
        )
        model = NormalizedGluReturnDensity(
            torch.zeros(120), torch.ones(120), density,
            return_count=4, widths=(8,), dropout=0, dropout_rate=0,
        )
        with torch.no_grad():
            model.prefix_conditioner[0, 0, 0] = 2.0
        features = torch.zeros((2, 120))
        targets = torch.tensor([
            [0.001, 0.0, 0.0, 0.0],
            [-0.001, 0.0, 0.0, 0.0],
        ])
        logits = model.teacher_forced_density_logits(features, targets)
        torch.testing.assert_close(logits[0, 0], logits[1, 0])
        self.assertFalse(torch.equal(logits[0, 1], logits[1, 1]))


if __name__ == "__main__":
    unittest.main()
