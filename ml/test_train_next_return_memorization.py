from __future__ import annotations

import unittest

import torch

from train_next_return_memorization import (
    adversarial_input_examples,
    adversarial_input_variant,
    adversarial_log_price_path_examples,
    adversarial_log_price_path_variant,
    adversarial_return_vector_examples,
    adversarial_return_vector_variant,
    gaussian_normalized_input_perturbation,
    mean_teacher_ema_decay,
    mean_teacher_ramp_multiplier,
    mean_teacher_variant,
    project_normalized_rms_l2,
    reconstruct_relative_log_price_path,
    sam_adversarial_return_vector_variant,
    sam_perturb_parameters,
    sam_restore_parameters,
    update_ema_teacher,
    validate_plan,
    weighted_robust_normalized_loss,
)


class RobustRegressionLossTest(unittest.TestCase):
    def test_huber_matches_mse_inlier_and_caps_outlier_gradient(self) -> None:
        prediction = torch.tensor([0.5, 4.0], requires_grad=True)
        loss = weighted_robust_normalized_loss(
            prediction,
            torch.zeros(2),
            torch.ones(2),
            1.0,
            {"type": "huber", "delta": 1.0},
        )
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), (0.25 + 7.0) / 2.0)
        self.assertAlmostEqual(float(prediction.grad[0]), 0.5)
        self.assertAlmostEqual(float(prediction.grad[1]), 1.0)

    def test_student_t_is_mse_equivalent_near_zero_and_suppresses_tail(self) -> None:
        prediction = torch.tensor([1e-3, 10.0])
        per_example = weighted_robust_normalized_loss(
            prediction,
            torch.zeros(2),
            torch.tensor([1.0, 0.0]),
            1.0,
            {"type": "student-t", "degreesOfFreedom": 3.0},
        )
        tail = weighted_robust_normalized_loss(
            prediction,
            torch.zeros(2),
            torch.tensor([0.0, 1.0]),
            1.0,
            {"type": "student-t", "degreesOfFreedom": 3.0},
        )
        self.assertAlmostEqual(float(per_example), 1e-6, places=9)
        self.assertLess(float(tail), 100.0)


class SamPerturbationTest(unittest.TestCase):
    def test_global_perturbation_has_rho_norm_and_restores_exactly(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 1),
        )
        original = tuple(value.detach().clone() for value in model.parameters())
        model(torch.ones(2, 3)).square().sum().backward()

        perturbations = sam_perturb_parameters(model, 0.05)  # type: ignore[arg-type]
        displacement = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(parameter.detach() - before)
            for parameter, before in zip(model.parameters(), original, strict=True)
        ]))
        self.assertAlmostEqual(float(displacement), 0.05, places=6)

        sam_restore_parameters(perturbations)
        for parameter, before in zip(model.parameters(), original, strict=True):
            self.assertTrue(torch.equal(parameter.detach(), before))


class _StandardizedLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("feature_std", torch.tensor([2.0, 0.5, 1.0]))
        self.weight = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return ((features / self.feature_std) * self.weight).sum(dim=1)


class _PathLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((120,), 0.01))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features * self.weight).sum(dim=1)


class MeanTeacherTest(unittest.TestCase):
    def test_half_life_decay_halves_a_parameter_after_selected_epochs(self) -> None:
        decay = mean_teacher_ema_decay(4.0, 16)
        self.assertAlmostEqual(decay ** (4 * 16), 0.5, places=12)

    def test_linear_ramp_uses_fraction_of_total_training(self) -> None:
        self.assertEqual(mean_teacher_ramp_multiplier(0, 100, 0.0), 1.0)
        self.assertEqual(mean_teacher_ramp_multiplier(0, 100, 0.5), 0.0)
        self.assertAlmostEqual(
            mean_teacher_ramp_multiplier(25, 100, 0.5), 0.5
        )
        self.assertEqual(mean_teacher_ramp_multiplier(50, 100, 0.5), 1.0)

    def test_gaussian_perturbation_has_exact_normalized_rms(self) -> None:
        model = _StandardizedLinear()
        features = torch.zeros(4, 3)
        attacked, normalized_noise = gaussian_normalized_input_perturbation(
            model, features, epsilon_rms=0.1, seed=7
        )
        self.assertTrue(torch.allclose(
            normalized_noise.square().mean(dim=1).sqrt(),
            torch.full((4,), 0.1),
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            (attacked - features) / model.feature_std,
            normalized_noise,
        ))
        attacked_again, _ = gaussian_normalized_input_perturbation(
            model, features, epsilon_rms=0.1, seed=7
        )
        self.assertTrue(torch.equal(attacked, attacked_again))

    def test_ema_update_moves_teacher_toward_student(self) -> None:
        teacher = torch.nn.Linear(2, 1, bias=False)
        student = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            teacher.weight.zero_()
            student.weight.fill_(2.0)
        update_ema_teacher(teacher, student, 0.75)
        self.assertTrue(torch.allclose(
            teacher.weight, torch.full_like(teacher.weight, 0.5)
        ))

    def test_variant_has_clean_mean_teacher_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = mean_teacher_variant(
            source,
            half_life_epochs=4,
            consistency_weight=1,
            ramp_up_fraction=0.05,
            input_perturbation_rms=0.01,
            suffix="mean-teacher-test-v1",
        )
        validate_plan(variant)
        self.assertEqual(variant["subset"]["examples"], 16)
        self.assertEqual(
            variant["training"]["meanTeacher"]["inferenceModel"],
            "ema-teacher",
        )
        self.assertEqual(
            variant["datasetFilter"]["type"],
            "exclude-exact-zero-target-return",
        )


class AdversarialInputTest(unittest.TestCase):
    def test_projection_bounds_normalized_rms_per_example(self) -> None:
        delta = torch.tensor([[3.0, 4.0], [0.01, 0.02]])
        projected = project_normalized_rms_l2(delta, 0.5)
        rms = projected.square().mean(dim=1).sqrt()
        self.assertAlmostEqual(float(rms[0]), 0.5, places=6)
        self.assertLessEqual(float(rms[1]), 0.5)

    def test_attack_maximizes_loss_in_normalized_input_ball(self) -> None:
        model = _StandardizedLinear()
        features = torch.zeros(2, 3)
        targets = torch.tensor([-1.0, 1.0])
        weights = torch.ones(2)
        before = features.clone()
        clean = (model(features) - targets).square().mean()

        attacked, normalized_delta = adversarial_input_examples(
            model,
            features,
            targets,
            weights,
            target_std=1.0,
            epsilon_rms=0.1,
            steps=1,
            step_size_rms=0.1,
        )

        self.assertTrue(torch.equal(features, before))
        self.assertTrue(torch.allclose(
            (attacked - features) / model.feature_std,
            normalized_delta,
        ))
        self.assertTrue(torch.allclose(
            normalized_delta.square().mean(dim=1).sqrt(),
            torch.full((2,), 0.1),
            atol=1e-6,
        ))
        self.assertGreater(
            float((model(attacked) - targets).square().mean().detach()),
            float(clean.detach()),
        )
        self.assertIsNone(model.weight.grad)

    def test_variant_has_valid_explicit_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = adversarial_input_variant(
            source,
            epsilon_rms=0.1,
            steps=2,
            adversarial_weight=0.5,
            suffix="adversarial-input-eps-1e-1-v1",
        )
        validate_plan(variant)
        self.assertEqual(
            variant["training"]["adversarialInput"]["stepSizeRms"], 0.05
        )


class AdversarialReturnVectorTest(unittest.TestCase):
    def test_attack_perturbs_inputs_and_target_inside_one_return_ball(self) -> None:
        model = _StandardizedLinear()
        features = torch.zeros(2, 3)
        targets = torch.tensor([-1.0, 1.0])
        weights = torch.ones(2)
        clean = (model(features) - targets).square().mean()

        attacked_features, attacked_targets, normalized_delta = (
            adversarial_return_vector_examples(
                model,
                features,
                targets,
                weights,
                target_std=1.0,
                epsilon_rms=0.1,
                steps=1,
                step_size_rms=0.1,
            )
        )

        self.assertEqual(tuple(normalized_delta.shape), (2, 4))
        self.assertTrue(torch.allclose(
            normalized_delta.square().mean(dim=1).sqrt(),
            torch.full((2,), 0.1),
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            (attacked_features - features) / model.feature_std,
            normalized_delta[:, :-1],
        ))
        self.assertTrue(torch.allclose(
            attacked_targets - targets,
            normalized_delta[:, -1],
        ))
        self.assertTrue(bool((normalized_delta[:, -1].abs() > 0).all()))
        self.assertGreater(
            float(
                (model(attacked_features) - attacked_targets)
                .square().mean().detach()
            ),
            float(clean.detach()),
        )
        self.assertIsNone(model.weight.grad)

    def test_variant_has_valid_input_output_return_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = adversarial_return_vector_variant(
            source,
            epsilon_rms=0.1,
            steps=2,
            adversarial_weight=0.5,
            suffix="adversarial-return-vector-eps-1e-1-v1",
        )
        validate_plan(variant)
        attack = variant["training"]["adversarialReturnVector"]
        self.assertEqual(attack["stepSizeRms"], 0.05)
        self.assertEqual(attack["returnCount"], 121)

    def test_combined_sam_variant_has_extended_budget_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = sam_adversarial_return_vector_variant(
            source,
            rho=1.0,
            epsilon_rms=0.01,
            steps=1,
            adversarial_weight=0.5,
            epochs=1024,
            suffix="sam-rho-1e0-return-eps-1e-2-1024epoch-v1",
        )
        validate_plan(variant)
        self.assertEqual(variant["training"]["epochs"], 1024)
        self.assertEqual(
            variant["training"]["sam"]["baseObjective"],
            "clean-plus-adversarial-return-vector",
        )
        self.assertTrue(
            variant["training"]["adversarialReturnVector"][
                "regenerateAtSamPerturbedWeights"
            ]
        )


class AdversarialLogPricePathTest(unittest.TestCase):
    def test_reconstructs_inputs_and_future_target_as_one_path(self) -> None:
        features = torch.arange(240, dtype=torch.float32).reshape(2, 120) / 100
        targets = torch.tensor([0.25, -0.5])
        path = reconstruct_relative_log_price_path(features, targets)
        returns = torch.diff(path, dim=1)

        self.assertEqual(tuple(path.shape), (2, 122))
        self.assertTrue(torch.allclose(returns[:, :-1], features))
        self.assertTrue(torch.allclose(returns[:, -1], targets))

    def test_attack_jointly_changes_inputs_and_target_inside_path_ball(self) -> None:
        model = _PathLinear()
        features = torch.zeros(2, 120)
        targets = torch.tensor([-1.0, 1.0])
        weights = torch.ones(2)
        clean = (model(features) - targets).square().mean()

        attacked_features, attacked_targets, normalized_path_delta = (
            adversarial_log_price_path_examples(
                model,
                features,
                targets,
                weights,
                target_std=1.0,
                epsilon_rms=0.1,
                steps=1,
                step_size_rms=0.1,
            )
        )

        self.assertTrue(torch.allclose(
            normalized_path_delta.square().mean(dim=1).sqrt(),
            torch.full((2,), 0.1),
            atol=1e-6,
        ))
        base_path = reconstruct_relative_log_price_path(features, targets)
        attacked_returns = torch.diff(
            base_path + normalized_path_delta, dim=1
        )
        self.assertTrue(torch.allclose(
            attacked_returns[:, :-1], attacked_features
        ))
        self.assertTrue(torch.allclose(
            attacked_returns[:, -1], attacked_targets
        ))
        self.assertFalse(torch.equal(attacked_targets, targets))
        self.assertGreater(
            float(
                (
                    model(attacked_features) - attacked_targets
                ).square().mean().detach()
            ),
            float(clean.detach()),
        )
        self.assertIsNone(model.weight.grad)

    def test_variant_has_valid_joint_path_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = adversarial_log_price_path_variant(
            source,
            epsilon_rms=0.1,
            steps=2,
            adversarial_weight=0.5,
            suffix="adversarial-log-price-eps-1e-1-v1",
        )
        validate_plan(variant)
        attack = variant["training"]["adversarialLogPricePath"]
        self.assertEqual(attack["stepSizeRms"], 0.05)
        self.assertEqual(attack["pathPoints"], 122)

    def test_fixed_future_attack_only_perturbs_observed_boundaries(self) -> None:
        model = _PathLinear()
        features = torch.zeros(2, 120)
        targets = torch.tensor([-1.0, 1.0])
        weights = torch.ones(2)
        base_path = reconstruct_relative_log_price_path(features, targets)

        attacked_features, attacked_targets, normalized_delta = (
            adversarial_log_price_path_examples(
                model,
                features,
                targets,
                weights,
                target_std=1.0,
                epsilon_rms=0.1,
                steps=1,
                step_size_rms=0.1,
                perturb_future_endpoint=False,
            )
        )

        self.assertEqual(tuple(normalized_delta.shape), (2, 121))
        attacked_path = torch.cat((
            base_path[:, :-1] + normalized_delta,
            base_path[:, -1:],
        ), dim=1)
        attacked_returns = torch.diff(attacked_path, dim=1)
        self.assertTrue(torch.allclose(
            attacked_returns[:, :-1], attacked_features
        ))
        self.assertTrue(torch.allclose(
            attacked_returns[:, -1], attacked_targets
        ))
        self.assertTrue(torch.equal(attacked_path[:, -1], base_path[:, -1]))
        self.assertTrue(torch.allclose(
            attacked_targets - targets,
            -normalized_delta[:, -1],
        ))

    def test_variant_has_valid_fixed_future_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = adversarial_log_price_path_variant(
            source,
            epsilon_rms=0.1,
            steps=1,
            adversarial_weight=0.5,
            perturb_future_endpoint=False,
            suffix="adversarial-observed-log-price-eps-1e-1-v1",
        )
        validate_plan(variant)
        attack = variant["training"]["adversarialLogPricePath"]
        self.assertEqual(attack["futureEndpoint"], "fixed-observed")
        self.assertEqual(attack["pathPoints"], 121)


if __name__ == "__main__":
    unittest.main()
