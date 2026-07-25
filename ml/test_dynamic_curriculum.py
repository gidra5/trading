from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_dynamic_curriculum import (
    collapse_equivalent_candidates,
    mean_policy_js_divergence,
    pareto_beam,
)


class DynamicCurriculumTests(unittest.TestCase):
    def test_policy_js_divergence_is_symmetric_and_zero_for_identity(self) -> None:
        left = np.asarray([[[0.2, 0.3, 0.5]]], dtype=np.float32)
        right = np.asarray([[[0.1, 0.2, 0.7]]], dtype=np.float32)

        self.assertEqual(mean_policy_js_divergence(left, left), 0)
        self.assertAlmostEqual(
            mean_policy_js_divergence(left, right),
            mean_policy_js_divergence(right, left),
        )
        self.assertGreater(mean_policy_js_divergence(left, right), 0)

    def test_equivalence_collapse_retains_the_lower_kl_representative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.candidate(root, "first", 0.5, [0.2, 0.3, 0.5])
            second = self.candidate(root, "second", 0.51, [0.2, 0.3, 0.5])
            distinct = self.candidate(root, "distinct", 0.52, [0.7, 0.2, 0.1])

            representatives, collapsed = collapse_equivalent_candidates(
                [second, distinct, first],
                {
                    "absoluteKl": 0.02,
                    "relativeKl": 0.01,
                    "maximumPolicyJsd": 1e-6,
                },
            )

            self.assertEqual(
                [candidate["key"] for candidate in representatives],
                ["first", "distinct"],
            )
            self.assertEqual(collapsed[0]["key"], "second")
            self.assertEqual(collapsed[0]["representative"], "first")

    def test_pareto_beam_uses_both_kl_mean_and_standard_deviation(self) -> None:
        candidates = [
            {"key": "low-mean", "validation": {
                "klDivergence": 0.5,
                "klDivergenceStdDev": 0.4,
            }},
            {"key": "low-spread", "validation": {
                "klDivergence": 0.6,
                "klDivergenceStdDev": 0.2,
            }},
            {"key": "dominated", "validation": {
                "klDivergence": 0.7,
                "klDivergenceStdDev": 0.5,
            }},
        ]

        selected, fronts = pareto_beam(candidates, 2)

        self.assertEqual(
            {candidate["key"] for candidate in selected},
            {"low-mean", "low-spread"},
        )
        self.assertEqual(set(fronts[0]), {"low-mean", "low-spread"})

    @staticmethod
    def candidate(
        root: Path,
        key: str,
        kl: float,
        probabilities: list[float],
    ) -> dict:
        directory = root / key
        directory.mkdir()
        value = np.asarray([[probabilities]], dtype="<f2")
        file = directory / "equivalence-signature.f16"
        file.write_bytes(value.tobytes())
        return {
            "key": key,
            "directory": str(directory),
            "validation": {
                "klDivergence": kl,
                "klDivergenceStdDev": 0.1,
            },
            "equivalenceSignature": {
                "file": file.name,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            },
        }


if __name__ == "__main__":
    unittest.main()
