from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_causal_oracle_predictability import mean_kl
from audit_oracle_policy_prototype_basis import (
    ACTION_COUNT,
    FitResult,
    centroid_update,
    deterministic_sample,
    fit_kl_prototypes,
    frank_wolfe_kl_projection,
    nearest_prototype_kl,
    write_prototype_artifact,
)
from trading_storage import read_shard_array


class Segment:
    def __init__(
        self,
        target_file: Path,
        target_row_offset: int,
        count: int,
    ) -> None:
        self.target_file = target_file
        self.target_row_offset = target_row_offset
        self.count = count


class PrototypeBasisAuditTest(unittest.TestCase):
    def test_deterministic_sample_spans_rows(self) -> None:
        values = np.arange(30, dtype=np.float32).reshape(10, 3)
        sampled = deterministic_sample(values, 4)
        np.testing.assert_array_equal(sampled, values[[0, 2, 5, 7]])

    def test_forward_kl_centroid_is_arithmetic_mean(self) -> None:
        values = np.asarray([
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.2, 0.6],
        ], dtype=np.float32)
        prototypes, counts = centroid_update(
            values,
            np.asarray([0, 0, 1, 1], dtype=np.int32),
            2,
        )
        np.testing.assert_array_equal(counts, [2, 2])
        np.testing.assert_allclose(
            prototypes,
            [[0.7, 0.15, 0.15], [0.15, 0.2, 0.65]],
            atol=1e-7,
        )

    def test_soft_projection_beats_nearest_without_hard_supervision(self) -> None:
        basis = np.asarray([
            [0.85, 0.10, 0.05],
            [0.05, 0.15, 0.80],
            [0.10, 0.80, 0.10],
        ], dtype=np.float32)
        weights = np.asarray([
            [0.5, 0.5, 0.0],
            [0.2, 0.3, 0.5],
            [0.0, 0.7, 0.3],
        ], dtype=np.float32)
        targets = weights @ basis
        nearest_kl, labels = nearest_prototype_kl(targets, basis)
        projected = frank_wolfe_kl_projection(
            targets,
            basis,
            initial_labels=labels,
            iterations=64,
            line_search_steps=18,
        )
        projected_kl = mean_kl(targets, projected["prediction"])
        self.assertGreater(nearest_kl, 0.05)
        self.assertLess(projected_kl, 2e-7)
        self.assertLess(projected["meanGap"], 2e-5)

    def test_fit_is_deterministic_and_prior_mixture_reconstructs_mean(self) -> None:
        rng = np.random.default_rng(17)
        targets = rng.dirichlet(np.ones(7), 320).astype(np.float32)
        first = fit_kl_prototypes(
            targets,
            8,
            sample_rows=192,
            lloyd_iterations=5,
            full_polish_passes=1,
            seed=41,
        )
        second = fit_kl_prototypes(
            targets,
            8,
            sample_rows=192,
            lloyd_iterations=5,
            full_polish_passes=1,
            seed=41,
        )
        np.testing.assert_array_equal(first.prototypes, second.prototypes)
        np.testing.assert_array_equal(first.train_weights, second.train_weights)
        np.testing.assert_allclose(
            first.train_weights @ first.prototypes,
            targets.mean(axis=0),
            atol=2e-7,
        )

    def test_artifact_is_content_addressed_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            source_reference = repo_root / "train-day.json"
            source_reference.write_text(
                '{"object":{"contentHash":"' + "a" * 64 + '"}}',
                encoding="utf-8",
            )
            values = np.full(
                (2, ACTION_COUNT),
                1 / ACTION_COUNT,
                dtype=np.float32,
            )
            values[1] = np.linspace(1, ACTION_COUNT, ACTION_COUNT)
            values[1] /= values[1].sum()
            fitted = FitResult(
                prototypes=values,
                train_weights=np.asarray([0.25, 0.75]),
                sample_objective=(0.2, 0.1),
                full_polish_objective=(0.08,),
                fit_sample_rows=4,
            )
            reference_file = (
                repo_root
                / "data/training/immutable/refs/models/example/basis-v1.json"
            )
            artifact = write_prototype_artifact(
                repo_root,
                reference_file,
                fitted,
                k=2,
                seed=7,
                lloyd_iterations=2,
                full_polish_passes=1,
                train_targets=values,
                train_segments=[Segment(source_reference, 0, 2)],
            )
            shard, decoded = read_shard_array(
                reference_file,
                "<f4",
                (2, ACTION_COUNT),
            )
            np.testing.assert_array_equal(decoded, values)
            expected_hash = hashlib.sha256(
                np.ascontiguousarray(values, dtype="<f4").tobytes()
            ).hexdigest()
            self.assertEqual(artifact["contentHash"], expected_hash)
            self.assertEqual(
                shard.reference["metadata"]["fitSplit"],
                "train",
            )
            self.assertFalse(
                shard.reference["metadata"]["validationUsedForFit"]
            )
            self.assertEqual(
                shard.reference["metadata"]["testPayloadsOpened"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
