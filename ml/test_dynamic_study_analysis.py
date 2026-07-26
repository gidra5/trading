from __future__ import annotations

import json
import unittest

import numpy as np

from dynamic_study_analysis import (
    LOSS_TERMS,
    analyze_epoch_response,
    greedy_d_optimal_indices,
    normalize_weights,
    parse_training_trajectories,
    response_features,
)


class DynamicStudyAnalysisTests(unittest.TestCase):
    def test_parses_completed_trajectory_and_ignores_interrupted_replacement(self) -> None:
        weights = {
            "cross_entropy": 1,
            "probability_mse": 0.1,
            "parameter_mse": 0.1,
            "excess_entropy": 0.1,
            "oracle_mutual_information": 1,
        }
        lines = [
            json.dumps({
                "event": "training-start",
                "predictionDelayMs": 60_000,
                "lossWeights": weights,
                "startEpoch": 0,
            }),
            json.dumps({
                "event": "epoch",
                "epoch": 0,
                "validation": {"klDivergence": 2.5},
            }),
            json.dumps({
                "event": "training-study-complete",
                "predictionDelayMs": 60_000,
                "lossWeights": weights,
            }),
            json.dumps({
                "event": "training-start",
                "predictionDelayMs": 60_000,
                "lossWeights": weights,
                "startEpoch": 0,
            }),
        ]

        trajectories = parse_training_trajectories(lines)

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].delay_ms, 60_000)
        self.assertEqual(trajectories[0].epochs[0]["klDivergence"], 2.5)

    def test_response_model_recovers_main_and_pairwise_surface(self) -> None:
        base = {
            "crossEntropy": 1,
            "probabilityMse": 0.1,
            "parameterMse": 0.1,
            "excessEntropy": 0.1,
            "oracleMutualInformation": 1,
        }
        coordinates = np.asarray([
            [
                1 if ((corner >> term) & 1) else -1
                for term in range(len(LOSS_TERMS))
            ]
            for corner in range(1 << len(LOSS_TERMS))
        ], dtype=np.float64)
        coordinates = np.vstack([np.zeros((1, len(LOSS_TERMS))), coordinates])
        features = response_features(coordinates)
        coefficients = np.linspace(-0.2, 0.3, features.shape[1])
        values = features @ coefficients
        observations = []
        for index, coordinate in enumerate(coordinates):
            weights = {
                term: base[term] * (4 ** coordinate[term_index])
                for term_index, term in enumerate(LOSS_TERMS)
            }
            observations.append({
                "variant": f"candidate-{index:02}",
                "weights": weights,
                "metrics": {"klDivergence": float(values[index])},
            })

        analysis = analyze_epoch_response(
            observations,
            base,
            probe_budgets=(16, 24, 32),
        )

        self.assertLess(analysis["fitRmse"], 1e-6)
        self.assertLess(analysis["crossValidatedRmse"], 1e-5)
        self.assertTrue(analysis["crossValidatedTop4Recall"])

    def test_d_optimal_selection_is_deterministic_and_unique(self) -> None:
        coordinates = np.asarray([
            [
                1 if ((corner >> term) & 1) else -1
                for term in range(len(LOSS_TERMS))
            ]
            for corner in range(32)
        ], dtype=np.float64)
        features = response_features(coordinates)

        first = greedy_d_optimal_indices(features, 24, required=(0,))
        second = greedy_d_optimal_indices(features, 24, required=(0,))

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(len(first), len(set(first)))

    def test_normalizes_python_loss_names(self) -> None:
        normalized = normalize_weights({
            "cross_entropy": 1,
            "probability_mse": 0.1,
            "parameter_mse": 0.1,
            "excess_entropy": 0.1,
            "oracle_mutual_information": 1,
        })

        self.assertIsNotNone(normalized)
        self.assertEqual(set(normalized or {}), set(LOSS_TERMS))


if __name__ == "__main__":
    unittest.main()
