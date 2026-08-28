from __future__ import annotations

import unittest

import torch

from compressed_path_return_density import CompressedPathOutput
from train_feature_compressed_path_density import (
    COMPONENT_LOGIT_OUTPUT_ADVERSARY,
    generate_output_adversarial_variant,
    output_adversarial_specification,
    perturb_component_logits,
)


class _OneStepModel:
    return_count = 1
    density_means = torch.tensor([-1.0, 0.0, 1.0])


def _output(logits: torch.Tensor) -> CompressedPathOutput:
    log_masses = torch.log_softmax(logits, dim=1)
    probabilities = torch.exp(log_masses)
    expectations = (
        probabilities * _OneStepModel.density_means[None, :]
    ).sum(dim=1, keepdim=True)
    return CompressedPathOutput(
        log_masses=(log_masses,),
        expectations=expectations,
    )


class OutputAdversarialPathDensityTest(unittest.TestCase):
    def test_specification_requires_coherent_clean_and_input_attack(self) -> None:
        specification = {
            "type": COMPONENT_LOGIT_OUTPUT_ADVERSARY,
            "epsilonRms": 0.01,
            "stepSizeRms": 0.01,
            "steps": 1,
            "adversarialWeight": 0.5,
            "applyTo": "clean-and-input-adversarial",
        }
        self.assertEqual(
            output_adversarial_specification({
                "adversarialOutput": specification
            }),
            specification,
        )

    def test_perturbation_renormalizes_and_recomputes_expectation(self) -> None:
        base = _output(torch.tensor([[0.2, -0.1, 0.4]]))
        delta = torch.tensor([[0.01, -0.02, 0.03]])
        perturbed = perturb_component_logits(
            base, (delta,), _OneStepModel()
        )
        torch.testing.assert_close(
            torch.exp(perturbed.log_masses[0]).sum(dim=1),
            torch.ones(1),
        )
        expected = (
            torch.exp(perturbed.log_masses[0])
            * _OneStepModel.density_means[None, :]
        ).sum(dim=1, keepdim=True)
        torch.testing.assert_close(perturbed.expectations, expected)

    def test_generated_attack_has_requested_rms_and_increases_loss(self) -> None:
        base = _output(torch.tensor([
            [0.2, -0.1, 0.4],
            [-0.2, 0.5, 0.1],
        ], requires_grad=True))

        def objective(candidate: CompressedPathOutput) -> torch.Tensor:
            return -candidate.log_masses[0][:, 0].mean()

        adversarial, metrics = generate_output_adversarial_variant(
            base,
            _OneStepModel(),
            objective,
            epsilon_rms=0.01,
            collect_metrics=True,
        )
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(
            metrics["normalizedLogitDeltaRms"], 0.01, places=6
        )
        self.assertEqual(
            metrics["finiteGradientCoordinates"],
            metrics["gradientCoordinates"],
        )
        self.assertGreaterEqual(
            float(objective(adversarial).detach()),
            float(objective(base).detach()),
        )


if __name__ == "__main__":
    unittest.main()
