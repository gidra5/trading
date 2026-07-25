from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mlp_model import ExposureMlp, INPUT_FEATURE_COUNT
from population_mlp import PopulationExposureMlp, population_clip_grad_norm_
from train_mlp_population import completed_job


class PopulationMlpTests(unittest.TestCase):
    def test_population_forward_matches_individual_model(self) -> None:
        torch.manual_seed(3)
        prototype = ExposureMlp(
            torch.linspace(-0.2, 0.2, INPUT_FEATURE_COUNT),
            torch.linspace(0.5, 1.5, INPUT_FEATURE_COUNT),
            dropout=0,
        ).eval()
        population = PopulationExposureMlp(prototype, 3).eval()
        features = torch.randn(5, INPUT_FEATURE_COUNT)

        expected = prototype(features)
        actual = population(features)

        self.assertEqual(actual.shape, (3, 5, 8))
        torch.testing.assert_close(actual[0], expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual[1], expected, rtol=1e-5, atol=1e-5)
        extracted = population.member_state_dict(2)
        self.assertEqual(extracted.keys(), prototype.state_dict().keys())
        for name, value in prototype.state_dict().items():
            torch.testing.assert_close(extracted[name], value)
        restored = ExposureMlp(
            extracted["feature_mean"],
            extracted["feature_std"],
            dropout=0,
        ).eval()
        restored.load_state_dict(extracted)
        torch.testing.assert_close(
            restored(features),
            actual[2],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_population_members_can_start_from_different_parent_states(self) -> None:
        torch.manual_seed(7)
        prototype = ExposureMlp(
            torch.zeros(INPUT_FEATURE_COUNT),
            torch.ones(INPUT_FEATURE_COUNT),
            dropout=0,
        ).eval()
        first = {
            name: value.detach().clone()
            for name, value in prototype.state_dict().items()
        }
        with torch.no_grad():
            for parameter in prototype.parameters():
                parameter.add_(0.05)
        second = {
            name: value.detach().clone()
            for name, value in prototype.state_dict().items()
        }
        population = PopulationExposureMlp(prototype, 2).eval()
        population.reset_from_state_dicts([first, second])
        features = torch.randn(3, INPUT_FEATURE_COUNT)
        first_model = ExposureMlp(
            first["feature_mean"],
            first["feature_std"],
            dropout=0,
        ).eval()
        first_model.load_state_dict(first)
        second_model = ExposureMlp(
            second["feature_mean"],
            second["feature_std"],
            dropout=0,
        ).eval()
        second_model.load_state_dict(second)

        actual = population(features)

        torch.testing.assert_close(actual[0], first_model(features))
        torch.testing.assert_close(actual[1], second_model(features))

    def test_population_gradient_clipping_is_per_member(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        parameter.grad = torch.tensor([[3.0, 4.0, 0.0], [0.3, 0.4, 0.0]])

        norms = population_clip_grad_norm_([parameter], 2, 1.0)

        torch.testing.assert_close(norms, torch.tensor([5.0, 0.5]))
        torch.testing.assert_close(
            parameter.grad,
            torch.tensor([[0.6, 0.8, 0.0], [0.3, 0.4, 0.0]]),
        )

    def test_multi_tensor_clipping_matches_concatenated_reference(self) -> None:
        torch.manual_seed(9)
        parameters = [
            torch.nn.Parameter(torch.zeros(3, 2, 4)),
            torch.nn.Parameter(torch.zeros(3, 5)),
            torch.nn.Parameter(torch.zeros(3, 2, 2)),
        ]
        original = []
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
            original.append(parameter.grad.clone())
        expected_norms = torch.stack([
            torch.cat([gradient[member].reshape(-1) for gradient in original]).norm()
            for member in range(3)
        ])

        actual_norms = population_clip_grad_norm_(parameters, 3, 0.75)

        torch.testing.assert_close(actual_norms, expected_norms)
        for member in range(3):
            actual = torch.cat([
                parameter.grad[member].reshape(-1) for parameter in parameters
            ])
            expected_scale = min(1.0, 0.75 / float(expected_norms[member]))
            expected = torch.cat([
                gradient[member].reshape(-1) for gradient in original
            ]) * expected_scale
            torch.testing.assert_close(actual, expected)

    def test_early_stopped_plateau_job_is_complete_for_same_contract(self) -> None:
        weights = {
            "crossEntropy": 1,
            "probabilityMse": 0.25,
            "parameterMse": 0,
            "excessEntropy": 0,
            "temporalMutualInformation": 4,
            "oracleMutualInformation": 1,
        }
        stored_weights = {
            "cross_entropy": 1.0,
            "probability_mse": 0.25,
            "parameter_mse": 0.0,
            "excess_entropy": 0.0,
            "temporal_mutual_information": 4.0,
            "oracle_mutual_information": 1.0,
        }
        common = {
            "epochs": 64,
            "patience": 6,
            "minimumImprovement": 0.001,
            "validationFraction": 0.25,
            "selectionMetric": "klDivergence",
            "seed": 1337,
        }
        manifest = {
            "planId": "plateau-plan",
            "predictionDelayMs": 3_599_000,
        }
        with TemporaryDirectory() as temporary:
            result_file = Path(temporary) / "study.json"
            result_file.write_text(json.dumps({
                "datasetPlanId": manifest["planId"],
                "predictionDelayMs": manifest["predictionDelayMs"],
                "selectionMetric": common["selectionMetric"],
                "epochs": common["epochs"],
                "epochsTrained": 9,
                "patience": common["patience"],
                "minimumImprovement": common["minimumImprovement"],
                "validationFraction": common["validationFraction"],
                "seed": common["seed"],
                "stoppedByPatience": True,
                "finalizedEarly": True,
                "lossWeights": stored_weights,
                "bestValidationMetrics": {"klDivergence": 0.5},
                "populationTraining": {"maxBatchesPerEpoch": None},
            }))
            job = {
                "resultFile": str(result_file),
                "output": temporary,
                "lossWeights": weights,
            }

            self.assertTrue(completed_job(job, manifest, common))
            changed = {**common, "minimumImprovement": 0.01}
            self.assertFalse(completed_job(job, manifest, changed))

if __name__ == "__main__":
    unittest.main()
