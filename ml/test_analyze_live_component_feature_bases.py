from __future__ import annotations

import unittest

import numpy as np

from ml.analyze_live_component_feature_bases import (
    availability_tie_key,
    components,
    joint_target,
    markdown_cell,
)


class LiveComponentFeatureBasisTest(unittest.TestCase):
    def test_component_contract_contains_all_declared_heads(self) -> None:
        definitions = components()
        self.assertEqual(len(definitions), 19)
        self.assertEqual(len({row.id for row in definitions}), 19)

    def test_joint_target_separates_zero_sign_and_magnitude(self) -> None:
        values = np.asarray([0.0, -0.5, -4.0, 0.5, 4.0])
        target = joint_target(values, np.asarray([1.0, 2.0, 3.0, 5.0]))
        np.testing.assert_array_equal(target, np.asarray([0, 1, 4, 5, 8]))

    def test_markdown_table_cells_escape_probability_condition_pipes(self) -> None:
        self.assertEqual(markdown_cell("P(|R| >= Q90 | active)"), "P(\\|R\\| >= Q90 \\| active)")

    def test_near_tie_prefers_the_subset_with_broader_availability(self) -> None:
        definitions = [
            {"availabilityScore": 0.45},
            {"availabilityScore": 0.95},
        ]
        scarce = {"indices": [0], "primaryBits": 0.101}
        broad = {"indices": [1], "primaryBits": 0.100}
        self.assertLess(availability_tie_key(broad, definitions), availability_tie_key(scarce, definitions))


if __name__ == "__main__":
    unittest.main()
