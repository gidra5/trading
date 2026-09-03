from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from hindsight_noise import (
    CONDITIONING_TYPE,
    TARGET_FREE_SCHEDULE_TYPE,
    advance_hindsight_curriculum,
    corrupt_standardized_hindsight,
    evaluation_noise_rng,
    expand_hindsight_samples,
    gaussian_hindsight,
    hindsight_variance_for_epoch,
    initial_hindsight_curriculum,
    noise_variance_at_epoch,
    validate_hindsight_plan,
)
from materialize_sparse_timeline_episode_dataset import (
    episode_lengths,
    sparse_episode_origins,
)
from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import (
    KnotDensityContract,
    ReturnTransform,
    component_log_masses,
)
from structured_feature_process import (
    CausalKnotBasisGramLossLayer8,
    CausalSelfAttentionLayer8,
    ConditionalEmbeddingBaseSupportGaussianMixture,
    ConditionalEmbeddingGaussianMixtureAttention,
    DualStateGatedExchangeCell,
    LowRankTensorPathGluBlock,
    StructuredSharedIoFeatureProcess,
)
from structured_union530_base import _safe_divide
from teacher_embedding_hindsight import (
    TeacherEmbeddingBatch, TeacherEmbeddingDataset, blend_embeddings, select_components,
)
from trading_storage import save_torch_checkpoint
from train_structured_feature_process import (
    COMPARABLE_EVALUATION_SCOPE,
    FeatureMetricAccumulator,
    FeatureSequenceMetricAccumulator,
    StructuredFeatureSequenceDataset,
    canonical_hash,
    checkpoint_payload,
    derive_base_support_feature_states,
    evaluate,
    evaluate_hindsight_curriculum_probe,
    evaluate_hindsight_reconstruction,
    production59_balanced_objective,
    run_epoch_limit,
    structured_feature_return_density_objective,
    target_free_continuation,
    validate_target_free_resume,
    weighted_standardized_mse,
)


class StructuredFeatureProcessTest(unittest.TestCase):
    def test_fixed_target_free_schedule_ignores_correlation_gate(self) -> None:
        schedule = {"type": TARGET_FREE_SCHEDULE_TYPE}
        self.assertIsNone(initial_hindsight_curriculum(schedule))
        for epoch in [0, 342, 373, 1000]:
            self.assertEqual(hindsight_variance_for_epoch(schedule, epoch, None), 1.0)

    def test_target_free_phase_is_pinned_bounded_and_resumable(self) -> None:
        plan = {"id": "phase-test", "training": {"epochs": 256}, "hindsightTeacher": {"frozen": True}}
        with TemporaryDirectory() as directory:
            root = Path(directory) / "data/training/runs/phase-test"
            (root / "state").mkdir(parents=True)
            self.assertIsNone(target_free_continuation(root, plan, 374))
            source = "checkpoints/milestones/before-target-free-e342.json"
            pointer = save_torch_checkpoint({"epoch": 341, "model": {"weight": torch.ones(1)}}, root / source)
            digest = pointer["object"]["contentHash"]
            phase = {"type": "fixed-target-free-continuation-v1", "planSha256": canonical_hash(plan),
                     "sourceCheckpoint": source, "sourceObjectSha256": digest,
                     "sourceCompletedEpochs": 342, "additionalEpochs": 32, "totalEpochs": 374,
                     "replacementFraction": 1.0}
            phase_file = root / "state/target-free-phase.json"
            phase_file.write_text(json.dumps(phase))
            self.assertEqual(target_free_continuation(root, plan, 374), phase)
            validate_target_free_resume(phase, {"epoch": 341}, digest)
            validate_target_free_resume(phase, {"epoch": 342, "targetFreePhase": phase}, "new-object")
            validate_target_free_resume(phase, {"epoch": 373, "targetFreePhase": phase}, "last-object")
            for saved, content_hash in [({"epoch": 340}, digest), ({"epoch": 341}, "wrong"),
                                         ({"epoch": 342}, digest), ({"epoch": 374, "targetFreePhase": phase}, digest)]:
                with self.subTest(saved=saved), self.assertRaises(ValueError):
                    validate_target_free_resume(phase, saved, content_hash)
            with self.assertRaises(ValueError):
                target_free_continuation(root, plan, 512)
            for change in [{"replacementFraction": .99}, {"sourceObjectSha256": "wrong"},
                           {"planSha256": "wrong"}, {"sourceCheckpoint": "../outside.json"},
                           {"additionalEpochs": 31}]:
                phase_file.write_text(json.dumps({**phase, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    target_free_continuation(root, plan, 374)

    def test_persisted_epoch_limit_preserves_plan_identity(self) -> None:
        plan = {"id": "same-run", "training": {"epochs": 256, "learningRate": 0.001}}
        original_hash = canonical_hash(plan)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(run_epoch_limit(root, plan), 256)
            (root / "state").mkdir()
            limit_file = root / "state/epoch-limit.json"
            limit_file.write_text(json.dumps({"planSha256": original_hash, "epochs": 512}))
            self.assertEqual(run_epoch_limit(root, plan), 512)
            self.assertEqual(canonical_hash(plan), original_hash)
            changed = {**plan, "training": {**plan["training"], "learningRate": 0.002}}
            with self.assertRaisesRegex(ValueError, "different training plan"):
                run_epoch_limit(root, changed)
            for invalid in [128, 0, True, 512.0, "512"]:
                limit_file.write_text(json.dumps({"planSha256": original_hash, "epochs": invalid}))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    run_epoch_limit(root, plan)

    def test_teacher_forecast_evaluation_uses_only_target_free_embeddings(self) -> None:
        model = self.make_model(2, 2, hindsight_conditioning={
            "sampleRepresentation": "frozen-teacher-component-embedding",
            "sampleInputWidth": 12, "sampleCount": 16, "sampleEmbeddingWidth": 256,
            "sampleCompression": "concatenated-affine-v1", "sampleObjective": "mean-per-sample-mse",
            "samplePrediction": "arithmetic-mean", "evaluationSeed": 19,
        })

        class Dataset(TeacherEmbeddingDataset):
            output_steps = 2

            def __init__(self):
                self.inputs = torch.randn(7, 2, 5)
                self.targets = torch.randn(7, 2, 5)
                self.calls = 0

            def iter_batches(self, split, batch_size, **kwargs):
                for start in range(0, 7, batch_size):
                    x, y = self.inputs[start:start + batch_size], self.targets[start:start + batch_size]
                    yield TeacherEmbeddingBatch(x, y, torch.ones(len(x)), {"past": x})

            def hindsight(self, inputs, context, *, targets, fraction, samples, rng):
                assert targets is None and fraction == 1 and samples == 16
                self.calls += 1
                # Deterministic inputs plus a distinct sampled component slot.
                z = torch.from_numpy(rng.random((len(inputs), 2, samples, 12), dtype=np.float32))
                return z + context["past"].mean(-1)[:, :, None, None]

        dataset = Dataset()
        predictions = []
        hook = model.register_forward_hook(lambda module, args, out: predictions.append(out.clone()))
        first = evaluate(model, dataset, "validation", batch_size=3, device=torch.device("cpu"))
        before = torch.cat(predictions)
        predictions.clear()
        dataset.targets += 100
        second = evaluate(model, dataset, "validation", batch_size=3, device=torch.device("cpu"))
        hook.remove()
        torch.testing.assert_close(before, torch.cat(predictions), rtol=0, atol=0)
        self.assertEqual(dataset.calls, 6)
        self.assertEqual(first["hindsightEvaluation"]["mode"], "one-pass-frozen-teacher-component-embeddings")
        self.assertNotEqual(first["nextReturn"]["mse"], second["nextReturn"]["mse"])

    def test_teacher_embeddings_compress_directly_without_feature_decoder(self) -> None:
        torch.manual_seed(919)
        model = self.make_model(2, 2, hindsight_conditioning={
            "sampleRepresentation": "frozen-teacher-component-embedding",
            "sampleInputWidth": 512, "sampleCount": 16,
            "sampleEmbeddingWidth": 256, "sampleCompression": "concatenated-affine-v1",
            "sampleObjective": "mean-per-sample-mse", "samplePrediction": "arithmetic-mean",
            "layer13Enabled": True, "layer13Count": 3, "layer13HiddenWidth": 32,
        })
        inputs = torch.randn(4, 2, 5)
        embeddings = torch.randn(4, 2, 16, 512)
        trace = model.trace(inputs, embeddings, torch.ones(4, 1))
        self.assertEqual(model.hindsight_embed.in_features, 16 * 512)
        self.assertEqual(model.hindsight_unembed.out_features, 16 * 5)
        self.assertEqual(trace.sample_outputs.shape, (4, 2, 16, 5))
        torch.testing.assert_close(trace.outputs, trace.sample_outputs.mean(2))
        trace.sample_outputs.square().mean().backward()
        for layer in (model.hindsight_embed, model.hindsight_unembed):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)
        changed = embeddings.clone()
        changed[:, 0] += 10
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, changed, torch.ones(4, 1))[:, 1], trace.outputs[:, 1])

    def test_teacher_replacement_endpoints_and_component_sampling(self) -> None:
        prior = torch.randn(3, 2, 16, 7)
        clean = torch.randn_like(prior, requires_grad=True)
        torch.testing.assert_close(blend_embeddings(clean, prior, 0), clean)
        torch.testing.assert_close(blend_embeddings(None, prior, 1), prior)
        torch.testing.assert_close(blend_embeddings(torch.full_like(prior, math.nan), prior, 1), prior)
        torch.testing.assert_close(blend_embeddings(clean, prior, .3), clean * .7 + prior * .3)
        self.assertFalse(blend_embeddings(clean, prior, .3).requires_grad)
        features = torch.tensor([[[[1., 2.], [3., 4.], [5., 6.]]]]).expand(7, 2, -1, -1)
        weights = torch.tensor([.2, .3, .5]).log().expand(7, 2, -1)
        uniforms = torch.from_numpy(np.random.default_rng(74).random((7, 2, 16), dtype=np.float32))
        whole = select_components(features, weights, uniforms)
        chunks = torch.cat([select_components(features[a:b], weights[a:b], uniforms[a:b])
                            for a, b in ((0, 3), (3, 7))])
        torch.testing.assert_close(whole, chunks, rtol=0, atol=0)
        for sample in whole.reshape(-1, 2):
            self.assertTrue(any(torch.equal(sample, atom) for atom in features[0, 0]))

    def test_deep_layer13_blocks_are_independent_and_reset_per_second(self) -> None:
        torch.manual_seed(213)
        model = self.make_model(2, 2, hindsight_conditioning={
            "layer13Enabled": True, "layer13HiddenWidth": 32, "layer13Count": 5,
        })
        blocks = (model.layer13, *model.layer13_refinements)
        self.assertEqual(len({id(p) for b in blocks for p in b.parameters()}),
                         sum(len(list(b.parameters())) for b in blocks))
        inputs, noise, variance = torch.randn(8, 2, 5), torch.randn(8, 2, 5), torch.ones(8, 1)
        trace = model.trace(inputs, noise, variance)
        self.assertEqual(len(trace.readout_residuals[0]), 6)
        for step in range(2):
            stream = trace.readout_updated_streams[step][0]
            for block in blocks:
                stream = stream + block(stream)
            torch.testing.assert_close(stream[:, 128:], trace.outputs[:, step])
            torch.testing.assert_close(trace.readout_initial_streams[step][:, :128],
                                       model.layer12.expand(8, -1))
        trace.outputs.square().mean().backward()
        for block in blocks:
            self.assertTrue(torch.isfinite(block.output.weight.grad).all())
            self.assertGreater(float(block.output.weight.grad.abs().sum()), 0)

    def test_multiple_samples_affine_compression_loss_and_gradient(self) -> None:
        for samples, layer13_count in ((4, 1), (16, 1), (16, 3)):
            torch.manual_seed(712)
            model = self.make_model(2, 2, hindsight_conditioning={
                "layer13Enabled": True, "layer13HiddenWidth": 32,
                "layer13Count": layer13_count,
                "sampleCount": samples, "sampleEmbeddingWidth": 256,
                "sampleCompression": "concatenated-affine-v1",
                "sampleObjective": "mean-per-sample-mse", "samplePrediction": "arithmetic-mean",
            })
            inputs, clean = torch.randn(8, 2, 5), torch.randn(8, 2, 5)
            targets = expand_hindsight_samples(clean, samples)
            noise, variance = gaussian_hindsight(np.random.default_rng(55), 8, 2, 5,
                                                torch.device("cpu"), samples=samples)
            self.assertFalse(torch.equal(noise[:, :, 0], noise[:, :, 1]))
            trace = model.trace(inputs, noise, variance)
            self.assertEqual(trace.sample_outputs.shape, (8, 2, samples, 5))
            self.assertEqual(trace.readout_initial_streams[0].shape, (8, 384))
            blocks = (model.layer13, *model.layer13_refinements)
            self.assertEqual(len(blocks), layer13_count)
            self.assertEqual(len({id(p) for b in blocks for p in b.parameters()}),
                             sum(len(list(b.parameters())) for b in blocks))
            for step in range(2):
                self.assertEqual(len(trace.readout_residuals[step]), 1 + layer13_count)
                torch.testing.assert_close(trace.readout_initial_streams[step][:, :128],
                                           model.layer12.expand(8, -1))
            torch.testing.assert_close(trace.outputs, trace.sample_outputs.mean(2))
            loss = weighted_standardized_mse(trace.sample_outputs, targets, torch.ones(8))
            torch.testing.assert_close(loss, (trace.sample_outputs - targets).square().mean())
            loss.backward()
            for module in (model.hindsight_embed, model.hindsight_unembed):
                self.assertTrue(torch.isfinite(module.weight.grad).all())
                self.assertGreater(float(module.weight.grad.abs().sum()), 0)
            for block in blocks:
                self.assertTrue(torch.isfinite(block.output.weight.grad).all())
                self.assertGreater(float(block.output.weight.grad.abs().sum()), 0)
            # Zero corrections preserve identical clean sample vectors exactly.
            with torch.no_grad():
                for block in (model.layer2, *blocks):
                    block.output.weight.zero_()
                    block.output.bias.zero_()
            torch.testing.assert_close(model.forward_samples(inputs, targets, variance * 0), targets)
            # A mean-only objective would incorrectly allow canceling errors.
            cancelling = targets.clone()
            cancelling[:, :, 0] += 1
            cancelling[:, :, 1] -= 1
            self.assertGreater(float(weighted_standardized_mse(cancelling, targets, torch.ones(8))), 0)
            torch.testing.assert_close(cancelling.mean(2), clean)
            # Seeded pure-noise evaluation is invariant to batch partitioning.
            rng = np.random.default_rng(55)
            chunked = torch.cat([gaussian_hindsight(rng, n, 2, 5, torch.device("cpu"),
                                samples=samples)[0] for n in (3, 5)])
            torch.testing.assert_close(chunked, noise, rtol=0, atol=0)

    def test_multisample_forecast_evaluation_never_receives_targets(self) -> None:
        torch.manual_seed(87)
        config = {
            "sampleCount": 4, "sampleEmbeddingWidth": 256,
            "evaluationSeed": 19,
            "sampleCompression": "concatenated-affine-v1",
            "sampleObjective": "mean-per-sample-mse", "samplePrediction": "arithmetic-mean",
        }
        model = self.make_model(2, 2, hindsight_conditioning=config)
        inputs = torch.randn(8, 2, 5)

        class Examples:
            channel_ids = tuple(f"feature-{i}" for i in range(5))
            output_steps = 2
            targets = torch.randn(8, 2, 5)

            def iter_batches(self, split, batch_size, **kwargs):
                for start in range(0, 8, batch_size):
                    end = start + batch_size
                    yield inputs[start:end], self.targets[start:end], torch.ones(len(inputs[start:end]))

        dataset = Examples()
        predictions = []
        handle = model.register_forward_hook(lambda module, args, output: predictions.append(output.clone()))
        first = evaluate(model, dataset, "validation", batch_size=3, device=torch.device("cpu"))
        before = torch.cat(predictions)
        predictions.clear()
        dataset.targets = dataset.targets * 7 + 9
        second = evaluate(model, dataset, "validation", batch_size=3, device=torch.device("cpu"))
        handle.remove()
        torch.testing.assert_close(before, torch.cat(predictions), rtol=0, atol=0)
        self.assertNotEqual(first["nextReturn"]["mse"], second["nextReturn"]["mse"])
        self.assertEqual(first["hindsightEvaluation"]["targetContribution"], 0)

    @staticmethod
    def gated_schedule() -> dict:
        return {
            "type": "correlation-gated-noise-variance-v1",
            "startEpoch": 0, "startVariance": 0, "endVariance": 1,
            "varianceIncrement": 0.01,
            "requiredCorrelation": 2 ** (-1 / 3600),
            "probeSplit": "train", "probeMetric": "nextReturn.correlation",
            "probeWeightSource": "raw-training-weights",
            "probeExamples": 65536, "probeSeed": 194,
        }

    def test_hindsight_gate_holds_until_exact_threshold_and_caps_at_one(self) -> None:
        schedule = self.gated_schedule()
        state = initial_hindsight_curriculum(schedule)
        threshold = schedule["requiredCorrelation"]
        for epoch, correlation in enumerate((None, math.nan, -0.2, threshold - 1e-10)):
            state = advance_hindsight_curriculum(
                schedule, state, epoch=epoch, correlation=correlation,
            )
            self.assertEqual(hindsight_variance_for_epoch(schedule, epoch + 1, state), 0)
        for epoch in range(4, 1004):
            state = advance_hindsight_curriculum(schedule, state, epoch=epoch, correlation=0.9)
        self.assertEqual(hindsight_variance_for_epoch(schedule, 1004, state), 0)
        state = advance_hindsight_curriculum(schedule, state, epoch=1004, correlation=threshold)
        self.assertEqual(hindsight_variance_for_epoch(schedule, 1005, state), 0.01)
        # Replaying an already completed epoch cannot silently increment twice.
        with self.assertRaisesRegex(ValueError, "epoch"):
            advance_hindsight_curriculum(schedule, state, epoch=1004, correlation=1)
        for epoch in range(1005, 1115):
            state = advance_hindsight_curriculum(schedule, state, epoch=epoch, correlation=1)
        self.assertEqual(hindsight_variance_for_epoch(schedule, 1115, state), 1)
        self.assertEqual(state["noiseStep"], 100)

    def test_hindsight_gate_is_saved_in_last_checkpoint_for_resume(self) -> None:
        schedule = self.gated_schedule()
        state = advance_hindsight_curriculum(
            schedule, initial_hindsight_curriculum(schedule), epoch=0, correlation=1,
        )
        model = self.make_model(2, 2, hindsight_conditioning={"evaluationSeed": 1})
        payload = checkpoint_payload(
            model, model.state_dict(), (), epoch=0, global_step=25,
            plan_hash="test", best={}, hindsight_curriculum=state,
        )
        restored = json.loads(json.dumps(payload["hindsightCurriculum"]))
        self.assertEqual(hindsight_variance_for_epoch(schedule, payload["epoch"] + 1, restored), 0.01)
        continued = advance_hindsight_curriculum(schedule, restored, epoch=1, correlation=0.99)
        self.assertEqual(hindsight_variance_for_epoch(schedule, 2, continued), 0.01)

    def test_hindsight_gate_probe_uses_training_targets_at_current_variance(self) -> None:
        torch.manual_seed(664)
        model = self.make_model(2, 2, hindsight_conditioning={"evaluationSeed": 1})
        inputs, targets = torch.randn(8, 2, 5), torch.randn(8, 2, 5)
        seen_splits = []

        class Examples:
            def iter_batches(self, split, batch_size, **kwargs):
                seen_splits.append(split)
                for start in range(0, len(inputs), batch_size):
                    end = start + batch_size
                    yield inputs[start:end], targets[start:end], torch.ones(len(inputs[start:end]))

        captured = []
        hook = model.register_forward_pre_hook(lambda module, args: captured.append(args[1].clone()))
        zero = evaluate_hindsight_curriculum_probe(
            model, Examples(), self.gated_schedule(), variance=0,
            batch_size=3, device=torch.device("cpu"),
        )
        torch.testing.assert_close(torch.cat(captured), model.standardized_targets(targets))
        self.assertEqual(zero["noiseVariance"], 0)
        self.assertEqual(zero["metric"], "nextReturn.correlation")
        self.assertEqual(zero["split"], "train")
        self.assertEqual(zero["examples"], 8)
        captured.clear()
        evaluate_hindsight_curriculum_probe(
            model, Examples(), self.gated_schedule(), variance=1,
            batch_size=3, device=torch.device("cpu"),
        )
        pure_noise = torch.cat(captured)
        captured.clear()
        targets += 5
        evaluate_hindsight_curriculum_probe(
            model, Examples(), self.gated_schedule(), variance=1,
            batch_size=3, device=torch.device("cpu"),
        )
        hook.remove()
        torch.testing.assert_close(pure_noise, torch.cat(captured), rtol=0, atol=0)
        self.assertEqual(seen_splits, ["train", "train", "train"])

    @staticmethod
    def hindsight_config(**overrides) -> dict:
        return {
            "type": CONDITIONING_TYPE, "injectionLayer": 2,
            "registerLayer": 12, "registerWidth": 128,
            "registerInitialization": "learned-zero-vector",
            "carryRegisters": False, "layer13Enabled": False,
            "sampleCount": 1, "sampleRepresentation": "full-output-feature-vector",
            "residualHiddenWidth": 512, "residualProjectionInitScale": 0.001,
            "residualPasses": 1,
            "residualParameterSharing": "across-forecast-steps-only",
            **overrides,
        }

    def make_model(
        self,
        input_steps: int,
        output_steps: int,
        *,
        linear_rank: int | None = None,
        layer8_attention: dict[str, int | str] | None = None,
        layer8_function_approximator: dict[str, object] | None = None,
        recurrent_memory: dict[str, object] | None = None,
        feature_embedding_density: dict[str, object] | None = None,
        return_density: dict[str, object] | None = None,
        return_density_contract: KnotDensityContract | None = None,
        recurrent_activation_checkpointing: bool = False,
        hindsight_conditioning: dict | None = None,
    ):
        if hindsight_conditioning is not None:
            hindsight_conditioning = self.hindsight_config(**hindsight_conditioning)
        return StructuredSharedIoFeatureProcess(
            torch.zeros(5),
            torch.ones(5),
            torch.zeros(5),
            torch.ones(5),
            input_steps=input_steps,
            output_steps=output_steps,
            feature_width=12,
            market_width=7,
            prefix_width=6,
            feature_distribution_width=8,
            extended_prefix_width=10,
            next_feature_distribution_width=9,
            initial_radius=0.0031622776601683794,
            minimum_radius=1e-4,
            learnable_centering=False,
            linear_rank=linear_rank,
            layer8_attention=layer8_attention,
            layer8_function_approximator=layer8_function_approximator,
            recurrent_memory=recurrent_memory,
            feature_embedding_density=feature_embedding_density,
            return_density=return_density,
            return_density_contract=return_density_contract,
            recurrent_activation_checkpointing=(
                recurrent_activation_checkpointing
            ),
            hindsight_conditioning=hindsight_conditioning,
        )

    def test_hindsight_variance_schedule_and_exact_pure_noise_endpoint(self) -> None:
        self.assertEqual(noise_variance_at_epoch(0, 48), 0.0)
        self.assertAlmostEqual(noise_variance_at_epoch(24, 48), 0.5)
        self.assertEqual(noise_variance_at_epoch(48, 48), 1.0)
        self.assertEqual(noise_variance_at_epoch(512, 48), 1.0)
        schedule = [noise_variance_at_epoch(epoch, 48) for epoch in range(50)]
        self.assertEqual(schedule, sorted(schedule))
        clean = torch.randn(7, 2, 5, requires_grad=True)
        noise = torch.randn_like(clean)
        torch.testing.assert_close(corrupt_standardized_hindsight(clean, noise, 0), clean)
        self.assertFalse(corrupt_standardized_hindsight(clean, noise, 0.3).requires_grad)
        torch.testing.assert_close(
            corrupt_standardized_hindsight(clean, noise, 0.3),
            math.sqrt(0.7) * clean + math.sqrt(0.3) * noise,
        )
        torch.testing.assert_close(
            corrupt_standardized_hindsight(torch.full_like(clean, math.nan), noise, 1),
            noise,
        )

    def test_hindsight_enters_only_matching_layer2_readout(self) -> None:
        torch.manual_seed(882)
        model = self.make_model(
            2, 2, hindsight_conditioning={
                "evaluationSeed": 7, "residualHiddenWidth": 512, "residualPasses": 2,
            },
            layer8_attention={
                "type": "causal-self-attention-v1", "heads": 1,
                "queryWidth": 12, "keyWidth": 12, "valueWidth": 12,
                "outputWidth": 12,
            },
        )
        inputs = torch.randn(7, 2, 5)
        noise = torch.randn(7, 2, 5)
        variance = torch.full((7, 1), 0.5)
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            model(inputs)
        prediction = model(inputs, noise, variance)
        self.assertEqual(model.layer2.output.out_features, 128 + 5)
        self.assertEqual(tuple(model.layer12.shape), (128,))
        self.assertIsNone(model.layer13)
        self.assertFalse(hasattr(model, "hindsight_projection"))
        self.assertEqual(prediction.shape, (7, 2, 5))
        changed_noise = noise.clone()
        changed_noise[:, 1] += 3
        changed_prediction = model(inputs, changed_noise, variance)
        torch.testing.assert_close(prediction[:, 0], changed_prediction[:, 0])
        self.assertFalse(torch.equal(prediction[:, 1], changed_prediction[:, 1]))
        changed_noise = noise.clone()
        changed_noise[:, 0] += 3
        changed_prediction = model(inputs, changed_noise, variance)
        # Unlike an early injection, changing step 1 cannot leak to step 2.
        torch.testing.assert_close(prediction[:, 1], changed_prediction[:, 1], rtol=0, atol=0)
        self.assertFalse(torch.equal(prediction[:, 0], changed_prediction[:, 0]))
        self.assertFalse(torch.equal(prediction, model(inputs, noise, torch.ones_like(variance))))
        baseline_trace = model.trace(inputs, noise, variance)
        changed_trace = model.trace(inputs, changed_noise, torch.ones_like(variance))
        for field in (
            "feature_embeddings", "market_states", "prefix_states",
            "feature_distribution_states", "extended_prefix_states",
            "next_feature_distribution_states", "expected_feature_embeddings",
        ):
            for before, after in zip(getattr(baseline_trace, field), getattr(changed_trace, field), strict=True):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        prediction.square().mean().backward()
        for parameter in model.layer2.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.norm()), 0)
        self.assertGreater(float(model.layer1.output.weight.grad.norm()), 0)
        self.assertGreater(float(model.layer12.grad.norm()), 0)
        self.assertTrue(bool(torch.isfinite(model.layer12.grad).all()))
        muon, adamw = optimizer_parameter_groups(model)
        routed = [id(value) for value in (*muon, *adamw)]
        self.assertEqual(len(routed), len(set(routed)))
        self.assertEqual(set(routed), {id(p) for p in model.parameters() if p.requires_grad})

    def test_layer2_starts_near_identity_without_forcing_noisy_copy(self) -> None:
        torch.manual_seed(771)
        model = self.make_model(2, 2, hindsight_conditioning={"evaluationSeed": 7})
        inputs = torch.randn(256, 2, 5)
        hindsight = torch.randn(256, 2, 5)
        prediction = model(inputs, hindsight, torch.zeros(256, 1))
        self.assertLess(float((prediction - hindsight).square().mean().detach()), 1e-5)
        correlation = float(torch.corrcoef(torch.stack((
            prediction[:, 0, 0].detach(), hindsight[:, 0, 0],
        )))[0, 1])
        self.assertGreaterEqual(correlation, 2 ** (-1 / 3600))
        self.assertTrue(model.layer12.requires_grad)
        # A zero residual is EXACT passthrough; the learned correction is free
        # to change the result without a learned multiplier on the bypass.
        with torch.no_grad():
            model.layer2.output.weight.zero_()
            model.layer2.output.bias.zero_()
        torch.testing.assert_close(
            model(inputs, hindsight, torch.ones(256, 1)),
            hindsight, rtol=0, atol=0,
        )
        with torch.no_grad():
            model.layer2.output.bias[128:].fill_(0.25)
        torch.testing.assert_close(
            model(inputs, hindsight, torch.ones(256, 1)), hindsight + 0.25,
            rtol=0, atol=0,
        )

    def test_layer2_hindsight_readout_restores_for_stopped_evaluation(self) -> None:
        from evaluate_stopped_structured_feature_process import model_from_state

        torch.manual_seed(224)
        model = self.make_model(2, 2, hindsight_conditioning={
            "evaluationSeed": 7, "residualHiddenWidth": 512, "residualPasses": 2,
        })
        architecture = {
            "inputSteps": 2, "outputSteps": 2, "featureWidth": 12,
            "marketWidth": 7, "prefixWidth": 6, "featureDistributionWidth": 8,
            "extendedPrefixWidth": 10, "nextFeatureDistributionWidth": 9,
            "initialRadius": 0.0031622776601683794, "minimumRadius": 1e-4,
            "learnableCentering": False,
            "hindsightConditioning": model.hindsight_conditioning,
        }
        with torch.no_grad():
            model.layer12.fill_(0.7)
            model.layer2.output.bias.fill_(0.2)
        restored = model_from_state(
            {"architecture": architecture}, model.state_dict(), torch.device("cpu"),
        )
        inputs, noise, variance = torch.randn(8, 2, 5), torch.randn(8, 2, 5), torch.ones(8, 1)
        torch.testing.assert_close(
            model(inputs, noise, variance), restored(inputs, noise, variance), rtol=0, atol=0,
        )

    def test_layer2_width_override_only_expands_readout_and_preserves_identity(self) -> None:
        torch.manual_seed(124)
        models = []
        for width, expected_parameters, layer13 in (
            (256, 4836212, False), (512, 6175092, False), (512, 5911924, True),
        ):
            config = self.hindsight_config(
                residualHiddenWidth=width, residualPasses=1 if layer13 else 2,
                layer13Enabled=layer13, layer13HiddenWidth=width,
            )
            model = StructuredSharedIoFeatureProcess(
                torch.zeros(59), torch.ones(59), torch.zeros(59), torch.ones(59),
                input_steps=2, output_steps=2, feature_width=256, market_width=128,
                prefix_width=128, feature_distribution_width=128,
                extended_prefix_width=256, next_feature_distribution_width=256,
                initial_radius=0.0031622776601683794, minimum_radius=1e-4,
                learnable_centering=False, hindsight_conditioning=config,
                layer8_attention={
                    "type": "causal-self-attention-v1", "heads": 1,
                    "queryWidth": 256, "keyWidth": 256, "valueWidth": 256,
                    "outputWidth": 256,
                },
            )
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected_parameters)
            self.assertEqual(model.layer2.output.in_features, width)
            self.assertEqual(model.layer2.projection.in_features, 444)
            self.assertEqual(model.layer2.output.out_features, 187)
            inputs, noise = torch.randn(128, 2, 59), torch.randn(128, 2, 59)
            with torch.no_grad():
                prediction = model(inputs, noise, torch.zeros(128, 1))
            self.assertLess(float((prediction - noise).square().mean()), 1e-5)
            self.assertGreater(float(torch.corrcoef(torch.stack((
                prediction[:, 0, 0], noise[:, 0, 0],
            )))[0, 1]), 2 ** (-1 / 3600))
            models.append(model)
        shapes = [
            {name: tuple(parameter.shape) for name, parameter in model.named_parameters()
             if not name.startswith(("layer2.", "layer2_refinements.", "layer13."))}
            for model in models
        ]
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[0], shapes[2])

    def test_layer13_receives_only_stream_and_reuses_weights_across_seconds(self) -> None:
        torch.manual_seed(369)
        config = {"layer13Enabled": True, "layer13HiddenWidth": 512}
        model = self.make_model(2, 2, hindsight_conditioning=config)
        self.assertEqual(len(model.layer2_refinements), 0)
        self.assertEqual(model.layer13.projection.in_features, 128 + 5)
        self.assertFalse({id(p) for p in model.layer2.parameters()} & {id(p) for p in model.layer13.parameters()})
        inputs, noise, variance = torch.randn(8, 2, 5), torch.randn(8, 2, 5), torch.ones(8, 1)
        seen = []
        hook = model.layer13.register_forward_pre_hook(lambda _, args: seen.append(args))
        trace = model.trace(inputs, noise, variance)
        hook.remove()
        self.assertEqual(len(seen), 2)
        manual_outputs = []
        for step in range(2):
            z0 = torch.cat((model.layer12.expand(8, -1), noise[:, step]), dim=-1)
            z1 = z0 + model.layer2(torch.cat((trace.expected_feature_embeddings[step], z0, variance), dim=-1))
            self.assertEqual(len(seen[step]), 1)
            torch.testing.assert_close(seen[step][0], z1, rtol=0, atol=0)
            torch.testing.assert_close(trace.readout_updated_streams[step][0], z1, rtol=0, atol=0)
            z2 = z1 + model.layer13(z1)
            torch.testing.assert_close(trace.readout_updated_streams[step][1], z2, rtol=0, atol=0)
            manual_outputs.append(z2[:, 128:])
        manual = torch.stack(manual_outputs, dim=1)
        torch.testing.assert_close(trace.outputs, manual, rtol=0, atol=0)
        params = (*model.layer2.parameters(), *model.layer13.parameters(), model.layer12)
        actual = torch.autograd.grad(trace.outputs.square().mean(), params, retain_graph=True)
        expected = torch.autograd.grad(manual.square().mean(), params)
        for a, b in zip(actual, expected, strict=True):
            self.assertTrue(bool(torch.isfinite(a).all()))
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-9)
        model(inputs, noise, variance).square().mean().backward()
        self.assertGreater(float(model.layer2.output.weight.grad[:128].norm()), 0)
        self.assertGreater(float(model.layer13.output.weight.grad[128:].norm()), 0)
        muon, adamw = optimizer_parameter_groups(model)
        self.assertTrue({id(p) for p in model.layer13.muon_parameters()} <= {id(p) for p in muon})
        routed = [id(p) for p in (*muon, *adamw)]
        self.assertEqual(len(routed), len(set(routed)))
        self.assertEqual(set(routed), {id(p) for p in model.parameters() if p.requires_grad})
        for changed_step in range(2):
            changed = noise.clone()
            changed[:, changed_step] += 2
            output = model(inputs, changed, variance)
            torch.testing.assert_close(output[:, 1 - changed_step], trace.outputs[:, 1 - changed_step], rtol=0, atol=0)
        with torch.no_grad():
            for layer in (model.layer2, model.layer13):
                layer.output.weight.zero_()
                layer.output.bias.zero_()
            torch.testing.assert_close(model(inputs, noise, variance), noise, rtol=0, atol=0)

    def test_layer13_checkpoint_restores_for_evaluation(self) -> None:
        from evaluate_stopped_structured_feature_process import model_from_state

        model = self.make_model(2, 2, hindsight_conditioning={
            "layer13Enabled": True, "layer13HiddenWidth": 512,
        })
        architecture = {
            "inputSteps": 2, "outputSteps": 2, "featureWidth": 12,
            "marketWidth": 7, "prefixWidth": 6, "featureDistributionWidth": 8,
            "extendedPrefixWidth": 10, "nextFeatureDistributionWidth": 9,
            "initialRadius": 0.0031622776601683794, "minimumRadius": 1e-4,
            "learnableCentering": False, "hindsightConditioning": model.hindsight_conditioning,
        }
        with torch.no_grad():
            model.layer13.output.bias.add_(0.125)
        restored = model_from_state({"architecture": architecture}, model.state_dict(), torch.device("cpu"))
        inputs, noise, variance = torch.randn(8, 2, 5), torch.randn(8, 2, 5), torch.ones(8, 1)
        torch.testing.assert_close(model(inputs, noise, variance), restored(inputs, noise, variance), rtol=0, atol=0)

    def test_residual_stream_matches_diagram_and_registers_reset_each_step(self) -> None:
        model = self.make_model(2, 2, hindsight_conditioning={})
        inputs, noise, variance = torch.randn(8, 2, 5), torch.randn(8, 2, 5), torch.ones(8, 1)
        with torch.no_grad():
            model.layer12.copy_(torch.linspace(-0.2, 0.2, 128))
        trace = model.trace(inputs, noise, variance)
        self.assertEqual(len(trace.readout_initial_streams), 2)
        for step in range(2):
            initial = torch.cat((model.layer12.expand(8, -1), noise[:, step]), dim=-1)
            residual = model.layer2(torch.cat((
                trace.expected_feature_embeddings[step], initial, variance,
            ), dim=-1))
            torch.testing.assert_close(trace.readout_initial_streams[step], initial, rtol=0, atol=0)
            torch.testing.assert_close(trace.readout_residuals[step][0], residual, rtol=0, atol=0)
            torch.testing.assert_close(trace.readout_updated_streams[step][0], initial + residual, rtol=0, atol=0)
            torch.testing.assert_close(trace.outputs[:, step], (initial + residual)[:, 128:], rtol=0, atol=0)
        trace.outputs.square().mean().backward()
        # With no layer 13 and no carry, register UPDATE rows are deliberately
        # unused. The learned INITIAL registers still influence sample updates.
        torch.testing.assert_close(model.layer2.output.weight.grad[:128], torch.zeros_like(model.layer2.output.weight.grad[:128]), rtol=0, atol=0)
        self.assertGreater(float(model.layer2.output.weight.grad[128:].norm()), 0)
        self.assertGreater(float(model.layer12.grad.norm()), 0)
        with torch.no_grad():
            model.layer2.output.bias[:128].add_(100)
        changed = model.trace(inputs, noise, variance)
        torch.testing.assert_close(changed.outputs, trace.outputs, rtol=0, atol=0)
        torch.testing.assert_close(changed.readout_initial_streams[1], trace.readout_initial_streams[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed.readout_updated_streams[0][0][:, :128], trace.readout_updated_streams[0][0][:, :128]))
        with torch.no_grad():
            model.layer12.add_(0.5)
        self.assertFalse(torch.equal(model(inputs, noise, variance), changed.outputs))

    def test_two_layer2_passes_are_independent_and_match_unrolled_gradients(self) -> None:
        torch.manual_seed(354)
        model = self.make_model(2, 2, hindsight_conditioning={"residualPasses": 2})
        torch.manual_seed(354)
        single_pass = self.make_model(2, 2, hindsight_conditioning={})
        # Extra depth does not perturb the existing modules' random draws.
        for name, value in single_pass.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(len(model.layer2_refinements), 1)
        layers = (model.layer2, model.layer2_refinements[0])
        self.assertFalse({id(p) for p in layers[0].parameters()} & {id(p) for p in layers[1].parameters()})
        self.assertFalse(torch.equal(layers[0].projection.weight, layers[1].projection.weight))
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in single_pass.parameters()) + sum(p.numel() for p in layers[1].parameters()),
        )
        muon, adamw = optimizer_parameter_groups(model)
        routed = [id(p) for p in (*muon, *adamw)]
        self.assertEqual(len(routed), len(set(routed)))
        self.assertEqual(set(routed), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertTrue({id(p) for p in layers[1].muon_parameters()} <= {id(p) for p in muon})
        inputs, noise, variance = torch.randn(8, 2, 5), torch.randn(8, 2, 5), torch.ones(8, 1)
        calls = []
        hooks = [layer.register_forward_pre_hook(
            lambda module, args, index=index: calls.append((index, args[0])),
        ) for index, layer in enumerate(layers)]
        trace = model.trace(inputs, noise, variance)
        for hook in hooks:
            hook.remove()
        # A then B for each second: no new module per forecast step.
        self.assertEqual([index for index, _ in calls], [0, 1] * model.output_steps)
        manual_outputs = []
        for step in range(2):
            initial = torch.cat((model.layer12.expand(8, -1), noise[:, step]), dim=-1)
            stream = initial
            self.assertEqual(len(trace.readout_residuals[step]), 2)
            self.assertEqual(len(trace.readout_updated_streams[step]), 2)
            torch.testing.assert_close(trace.readout_initial_streams[step], initial, rtol=0, atol=0)
            for pass_index in range(2):
                expected_input = torch.cat((trace.expected_feature_embeddings[step], stream, variance), dim=-1)
                torch.testing.assert_close(calls[step * 2 + pass_index][1], expected_input, rtol=0, atol=0)
                residual = layers[pass_index](expected_input)
                stream = stream + residual
                torch.testing.assert_close(trace.readout_residuals[step][pass_index], residual, rtol=0, atol=0)
                torch.testing.assert_close(trace.readout_updated_streams[step][pass_index], stream, rtol=0, atol=0)
            manual_outputs.append(stream[:, 128:])
        manual = torch.stack(manual_outputs, dim=1)
        torch.testing.assert_close(trace.outputs, manual, rtol=0, atol=0)
        parameters = tuple(p for layer in layers for p in layer.parameters()) + (model.layer12,)
        actual_grads = torch.autograd.grad(trace.outputs.square().mean(), parameters, retain_graph=True)
        manual_grads = torch.autograd.grad(manual.square().mean(), parameters)
        for actual, expected in zip(actual_grads, manual_grads, strict=True):
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9)
        model.zero_grad(set_to_none=True)
        model(inputs, noise, variance).square().mean().backward()
        # Pass two reads pass one's updated registers, enabling their gradients.
        self.assertGreater(float(model.layer2.output.weight.grad[:128].norm()), 0)
        self.assertGreater(float(model.layer2.output.weight.grad[128:].norm()), 0)
        self.assertGreater(float(layers[1].output.weight.grad[128:].norm()), 0)
        # The final register update is discarded; only A's register update is read.
        torch.testing.assert_close(layers[1].output.weight.grad[:128], torch.zeros_like(layers[1].output.weight.grad[:128]), rtol=0, atol=0)
        self.assertGreater(float(model.layer12.grad.norm()), 0)
        with torch.no_grad():
            model.layer2.output.bias[:128].add_(100)
        changed = model.trace(inputs, noise, variance)
        self.assertFalse(torch.equal(changed.outputs, trace.outputs))
        torch.testing.assert_close(changed.readout_initial_streams[1], trace.readout_initial_streams[1], rtol=0, atol=0)
        with torch.no_grad():
            layers[1].output.bias[128:].add_(0.125)
        changed_b = model.trace(inputs, noise, variance)
        for step in range(2):
            torch.testing.assert_close(changed_b.readout_updated_streams[step][0], changed.readout_updated_streams[step][0], rtol=0, atol=0)
            self.assertFalse(torch.equal(changed_b.outputs[:, step], changed.outputs[:, step]))

    def test_two_layer2_passes_preserve_near_identity_and_literal_bypass(self) -> None:
        torch.manual_seed(355)
        model = self.make_model(2, 2, hindsight_conditioning={"residualPasses": 2})
        inputs, noise, variance = torch.randn(256, 2, 5), torch.randn(256, 2, 5), torch.zeros(256, 1)
        with torch.no_grad():
            output = model(inputs, noise, variance)
            self.assertLess(float((output - noise).square().mean()), 1e-5)
            self.assertGreaterEqual(float(torch.corrcoef(torch.stack((
                output[:, 0, 0], noise[:, 0, 0],
            )))[0, 1]), 2 ** (-1 / 3600))
            for layer in (model.layer2, *model.layer2_refinements):
                layer.output.weight.zero_()
                layer.output.bias.zero_()
            torch.testing.assert_close(model(inputs, noise, variance), noise, rtol=0, atol=0)
            model.layer2.output.bias[128:].fill_(0.25)
            model.layer2_refinements[0].output.bias[128:].fill_(0.5)
            # Independently learned corrections; no sample reset between passes.
            torch.testing.assert_close(model(inputs, noise, variance), (noise + 0.25) + 0.5, rtol=0, atol=0)

    def test_held_out_reconstruction_is_separate_from_training_gate(self) -> None:
        model = self.make_model(2, 2, hindsight_conditioning={"evaluationSeed": 19})
        inputs, targets = torch.randn(8, 2, 5), torch.randn(8, 2, 5)
        seen_splits = []

        class Examples:
            def iter_batches(self, split, batch_size, **kwargs):
                seen_splits.append(split)
                for start in range(0, len(inputs), batch_size):
                    end = min(start + batch_size, len(inputs))
                    yield inputs[start:end], targets[start:end], torch.ones(end - start)

        schedule = self.gated_schedule()
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        train = evaluate_hindsight_curriculum_probe(
            model, Examples(), schedule, variance=0.01,
            batch_size=3, device=torch.device("cpu"),
        )
        validation = evaluate_hindsight_reconstruction(
            model, Examples(), schedule, split="validation", variance=0.01,
            batch_size=3, device=torch.device("cpu"),
        )
        self.assertEqual(seen_splits, ["train", "validation"])
        self.assertTrue(train["usedForCurriculum"])
        self.assertFalse(validation["usedForCurriculum"])
        self.assertFalse(validation["usedForCheckpointSelection"])
        self.assertEqual(validation["noiseSeed"], train["noiseSeed"] + 1)
        self.assertEqual(validation["noiseVariance"], train["noiseVariance"])
        self.assertEqual(len(validation["perStepNextReturn"]), 2)
        self.assertTrue(math.isfinite(validation["mseSkillVsZero"]))
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        for name, before in state_before.items():
            torch.testing.assert_close(model.state_dict()[name], before, rtol=0, atol=0)

    def test_hindsight_evaluation_predictions_do_not_depend_on_targets(self) -> None:
        torch.manual_seed(194)
        model = self.make_model(2, 2, hindsight_conditioning={"evaluationSeed": 19})
        inputs = torch.randn(8, 2, 5)

        class Examples:
            output_steps = 2
            targets = torch.randn(8, 2, 5)

            def iter_batches(self, split, batch_size, **kwargs):
                for start in range(0, len(inputs), batch_size):
                    end = start + batch_size
                    yield inputs[start:end], self.targets[start:end], torch.ones(len(inputs[start:end]))

        dataset = Examples()
        recorded = []
        handle = model.register_forward_hook(lambda module, args, output: recorded.append(output.clone()))
        first = evaluate(model, dataset, "validation", batch_size=3, device=torch.device("cpu"))
        before = torch.cat(recorded)
        recorded.clear()
        dataset.targets = 5 * torch.randn_like(dataset.targets) + 2
        second = evaluate(model, dataset, "validation", batch_size=3, device=torch.device("cpu"))
        handle.remove()
        torch.testing.assert_close(before, torch.cat(recorded), rtol=0, atol=0)
        self.assertNotEqual(first["nextReturn"]["mse"], second["nextReturn"]["mse"])
        self.assertEqual(first["hindsightEvaluation"]["targetContribution"], 0)

    def test_evaluation_noise_is_split_seeded_and_batch_independent(self) -> None:
        config = {"evaluationSeed": 29}
        rng = evaluation_noise_rng(config, "validation")
        chunked = torch.cat([
            gaussian_hindsight(rng, size, 2, 5, torch.device("cpu"))[0]
            for size in (3, 4, 1)
        ])
        full, variance = gaussian_hindsight(
            evaluation_noise_rng(config, "validation"), 8, 2, 5, torch.device("cpu"),
        )
        torch.testing.assert_close(chunked, full, rtol=0, atol=0)
        self.assertTrue(bool((variance == 1).all()))
        test, _ = gaussian_hindsight(
            evaluation_noise_rng(config, "test"), 8, 2, 5, torch.device("cpu"),
        )
        self.assertFalse(torch.equal(full, test))

    def test_hindsight_plan_rejects_target_assisted_evaluation(self) -> None:
        plan = {
            "architecture": {"hindsightConditioning": self.hindsight_config(
                evaluationVariance=1.0, evaluationSeed=29,
            )},
            "training": {
                "epochs": 512,
                "loss": {"type": "mean-standardized-next-feature-mse-v1"},
                "hindsightNoiseSchedule": {
                    "type": "cosine-noise-variance-v1", "startEpoch": 0,
                    "startVariance": 0, "endVariance": 1, "endEpoch": 48,
                },
            },
        }
        validate_hindsight_plan(plan)
        for width in (0, -1, 512.5, True):
            plan["architecture"]["hindsightConditioning"]["residualHiddenWidth"] = width
            with self.assertRaisesRegex(ValueError, "hidden widths"):
                validate_hindsight_plan(plan)
        plan["architecture"]["hindsightConditioning"]["residualHiddenWidth"] = 512
        validate_hindsight_plan(plan)
        config = plan["architecture"]["hindsightConditioning"]
        config["residualPasses"] = 2
        validate_hindsight_plan(plan)
        config["residualParameterSharing"] = "shared-across-passes"
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_hindsight_plan(plan)
        config["residualParameterSharing"] = "across-forecast-steps-only"
        for bad_passes in (0, -1, 2.5, True):
            config["residualPasses"] = bad_passes
            with self.assertRaisesRegex(ValueError, "residual passes"):
                validate_hindsight_plan(plan)
        config["residualPasses"] = 2
        for key, bad_value in (("carryRegisters", True), ("layer13Enabled", True),
                               ("sampleCount", 2), ("sampleCount", True),
                               ("sampleRepresentation", "scalar-return"),
                               ("registerWidth", 0), ("residualProjectionInitScale", 0)):
            valid_value = config[key]
            config[key] = bad_value
            with self.assertRaises(ValueError):
                validate_hindsight_plan(plan)
            config[key] = valid_value
        config["residualPasses"] = 1
        config["layer13Enabled"] = True
        config["layer13HiddenWidth"] = 512
        validate_hindsight_plan(plan)
        for width in (0, -1, True, 512.5):
            config["layer13HiddenWidth"] = width
            with self.assertRaisesRegex(ValueError, "layer 13 hidden width"):
                validate_hindsight_plan(plan)
        config["layer13HiddenWidth"] = 512
        plan["training"]["hindsightNoiseSchedule"] = self.gated_schedule()
        validate_hindsight_plan(plan)
        plan["training"]["hindsightNoiseSchedule"]["probeSplit"] = "validation"
        with self.assertRaisesRegex(ValueError, "correlation gate"):
            validate_hindsight_plan(plan)
        plan["training"]["hindsightNoiseSchedule"]["probeSplit"] = "train"
        plan["architecture"]["hindsightConditioning"]["evaluationVariance"] = 0.99
        with self.assertRaisesRegex(ValueError, "target-free endpoint"):
            validate_hindsight_plan(plan)

    def test_scalar_return_density_reads_expected_feature_embedding(
        self,
    ) -> None:
        density = {
            "type": "fixed-knot-piecewise-linear-return-density-v1",
            "latentSource": "expected-feature-embedding",
            "knotCount": 5,
        }
        torch.manual_seed(107)
        model = self.make_model(1, 1, return_density=density)
        areas = torch.tensor([0.1, 0.2, 0.25, 0.3, 0.15])
        masses = torch.tensor([0.05, 0.15, 0.4, 0.3, 0.1])
        model.initialize_return_density_prior(areas, masses)
        inputs = torch.randn(7, 1, 5)
        predictions, logits = model.forward_with_return_density(inputs)
        self.assertEqual(predictions.shape, (7, 1, 5))
        self.assertEqual(logits.shape, (7, 1, 5))
        torch.testing.assert_close(
            component_log_masses(logits, areas),
            masses.log().expand(7, 1, 5),
        )
        loss = predictions.square().mean() + logits.square().mean()
        loss.backward()
        for parameter in model.return_density_head.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_structured_feature_return_density_objective_is_equal_weight(
        self,
    ) -> None:
        feature_mse = torch.tensor(2.0, requires_grad=True)
        return_nll = torch.tensor(-8.0, requires_grad=True)
        components = structured_feature_return_density_objective(
            feature_mse,
            return_nll,
            {
                "featureObjective": (
                    "mean-standardized-derived-production59-feature-mse-v1"
                ),
                "weights": {
                    "derivedFeatureMse": 0.5,
                    "returnDensityNll": 0.5,
                },
            },
        )
        self.assertEqual(float(components["objective"].detach()), -3.0)
        components["objective"].backward()
        self.assertEqual(float(feature_mse.grad), 0.5)
        self.assertEqual(float(return_nll.grad), 0.5)

    def test_joint_path_return_density_emits_exact_conditional_terms(
        self,
    ) -> None:
        contract = KnotDensityContract(
            transform=ReturnTransform(
                alpha=2.0, location_bps=0.0, scale_bps=1.0,
            ),
            knots_unit=np.linspace(0.0, 1.0, 5),
            prior_component_masses=np.full(5, 0.2),
            source_file="test",
            source_fit="5",
        )
        density = {
            "type": (
                "joint-prefix-contracted-path-matrix-return-density-v1"
            ),
            "latentSource": "expected-feature-embedding",
            "knotCount": 5,
            "returnCount": 3,
            "marketWidth": 12,
            "pathEmbeddingWidth": 4,
            "pathCount": 5,
            "stageBlockCount": 1,
            "pathCompressionWidth": 6,
            "jointCompressionWidth": 7,
            "recurrentActivationCheckpointing": True,
            "initialRadius": 0.0031622776601683794,
            "minimumRadius": 1e-4,
            "learnableCentering": False,
        }
        torch.manual_seed(109)
        model = self.make_model(
            1,
            3,
            return_density=density,
            return_density_contract=contract,
            recurrent_activation_checkpointing=True,
        )
        inputs = torch.randn(4, 1, 5)
        targets = torch.randn(4, 3) * 5e-5
        prediction, output = model.forward_with_joint_return_density(
            inputs, targets
        )
        self.assertEqual(prediction.shape, (4, 3, 5))
        self.assertEqual(output.expectations.shape, (4, 3))
        self.assertIsNotNone(output.joint_log_density_terms)
        self.assertEqual(output.joint_log_density_terms.shape, (4, 3))
        loss = prediction.square().mean() \
            - output.joint_log_density_terms.mean()
        loss.backward()
        self.assertIsNotNone(model.layer1.output.weight.grad)
        for parameter in model.return_density_head.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_checkpointed_rollout_matches_unrolled_gradients(self) -> None:
        torch.manual_seed(113)
        plain = self.make_model(1, 4)
        checkpointed = self.make_model(
            1, 4, recurrent_activation_checkpointing=True
        )
        checkpointed.load_state_dict(plain.state_dict())
        plain.train()
        checkpointed.train()
        inputs = torch.randn(5, 1, 5)
        plain_output = plain(inputs)
        checkpointed_output = checkpointed(inputs)
        torch.testing.assert_close(plain_output, checkpointed_output)
        plain_output.square().mean().backward()
        checkpointed_output.square().mean().backward()
        plain_gradients = dict(plain.named_parameters())
        for name, parameter in checkpointed.named_parameters():
            torch.testing.assert_close(
                plain_gradients[name].grad,
                parameter.grad,
                rtol=2e-5,
                atol=2e-6,
            )

    def test_feature_embedding_attention_is_normalized_mixture_nll(
        self,
    ) -> None:
        attention = {
            "type": "causal-self-attention-v1",
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
        }
        density = {
            "type": "conditional-gaussian-mixture-attention-v1",
            "components": 4,
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 1,
            "fixedStandardDeviation": 2.0,
            "normalizationEpsilon": 1e-8,
            "queryNormalization": "unit-rms",
            "targetEncoderGradient": "stop-gradient",
        }
        torch.manual_seed(101)
        model = self.make_model(
            2, 2,
            layer8_attention=attention,
            feature_embedding_density=density,
        )
        self.assertIsInstance(
            model.embedding_density_attention,
            ConditionalEmbeddingGaussianMixtureAttention,
        )
        inputs = torch.randn(7, 2, 5)
        targets = torch.randn(7, 2, 5, requires_grad=True)
        log_density = model.feature_embedding_log_density(inputs, targets)
        self.assertEqual(log_density.shape, (7, 2))
        self.assertTrue(bool(torch.isfinite(log_density).all()))
        (-log_density.mean()).backward()
        self.assertIsNone(targets.grad)
        for parameter in model.embedding_density_attention.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        for parameter in model.layer8.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_base_support_density_has_exact_base_expectation_and_nll(
        self,
    ) -> None:
        attention = {
            "type": "causal-self-attention-v1",
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
        }
        density = {
            "type": "causal-base-support-gaussian-mixture-density-v2",
            "components": 4,
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 5,
            "fixedStandardDeviation": 2.0,
            "normalizationEpsilon": 1e-8,
            "queryNormalization": "unit-rms",
            "targetEncoderGradient": "stop-gradient",
            "keyConstruction": (
                "causal-derived-feature-state-through-shared-layer1"
            ),
        }
        torch.manual_seed(103)
        model = self.make_model(
            2, 2,
            layer8_attention=attention,
            feature_embedding_density=density,
        )
        self.assertIsInstance(
            model.embedding_density_attention,
            ConditionalEmbeddingBaseSupportGaussianMixture,
        )
        inputs = torch.randn(7, 2, 5)
        targets = torch.randn(7, 2, 5, requires_grad=True)
        raw_components, log_weights = \
            model.feature_embedding_base_distribution(inputs)
        self.assertEqual(raw_components.shape, (7, 2, 4, 5))
        self.assertEqual(log_weights.shape, (7, 2, 4))
        torch.testing.assert_close(
            torch.logsumexp(log_weights, dim=-1),
            torch.zeros(7, 2),
        )
        log_density = model.feature_embedding_log_density_from_base_support(
            targets, raw_components, log_weights,
        )
        expected = model.expected_raw_base_features(
            raw_components, log_weights,
        )
        torch.testing.assert_close(
            expected,
            (log_weights.exp().unsqueeze(-1) * raw_components).sum(dim=-2),
        )
        self.assertTrue(bool(torch.isfinite(log_density).all()))
        (-log_density.mean()).backward()
        self.assertIsNone(targets.grad)
        for parameter in model.embedding_density_attention.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        for parameter in model.layer8.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_base_support_paths_are_rederived_component_by_component(
        self,
    ) -> None:
        class ToyDataset:
            output_steps = 2
            base_coordinate_count = 5
            feature_count = 5

            @staticmethod
            def derive(base, inputs, context):
                del inputs
                return base + context["offset"][:, None]

        raw_components = torch.arange(
            3 * 2 * 4 * 5, dtype=torch.float32,
        ).reshape(3, 2, 4, 5)
        inputs = torch.zeros(3, 2, 5)
        offset = torch.tensor([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ], dtype=torch.float32)
        derived = derive_base_support_feature_states(
            ToyDataset(), raw_components, inputs, {"offset": offset},
        )
        torch.testing.assert_close(
            derived, raw_components + offset[:, None, None],
        )

    def test_feature_embedding_density_rejects_recurrent_layer_state(
        self,
    ) -> None:
        density = {
            "type": "conditional-gaussian-mixture-attention-v1",
            "components": 4,
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 1,
            "fixedStandardDeviation": 2.0,
            "normalizationEpsilon": 1e-8,
            "queryNormalization": "unit-rms",
            "targetEncoderGradient": "stop-gradient",
        }
        recurrent = {
            "type": "dual-state-gated-exchange-v1",
            "hiddenWidth": 4,
            "layers": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
            "activation": "sigmoid",
            "mix": "a*t+b*(t-1)",
        }
        with self.assertRaisesRegex(ValueError, "requires stateless"):
            self.make_model(
                2, 2,
                recurrent_memory=recurrent,
                feature_embedding_density=density,
            )

    def test_layer8_causal_attention_uses_prefix_nf_states(self) -> None:
        attention = {
            "type": "causal-self-attention-v1",
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
        }
        torch.manual_seed(23)
        one_step = self.make_model(2, 1, layer8_attention=attention)
        two_step_model = self.make_model(2, 2, layer8_attention=attention)
        two_step_model.load_state_dict(one_step.state_dict())
        self.assertIsInstance(
            two_step_model.layer8, CausalSelfAttentionLayer8
        )

        inputs = torch.randn(7, 2, 5)
        first = one_step.trace(inputs)
        second = two_step_model.trace(inputs)
        torch.testing.assert_close(
            first.expected_feature_embeddings[0],
            second.expected_feature_embeddings[0],
        )
        self.assertEqual(second.outputs.shape, (7, 2, 5))

        loss = second.outputs.square().mean()
        loss.backward()
        for parameter in two_step_model.layer8.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_every_repeated_gnglu_layer_carries_private_hidden_memory(
        self,
    ) -> None:
        attention = {
            "type": "causal-self-attention-v1",
            "queryWidth": 12,
            "keyWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
        }
        memory = {
            "type": "dual-state-gated-exchange-v1",
            "hiddenWidth": 4,
            "layers": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
            "activation": "sigmoid",
            "mix": "a*t+b*(t-1)",
        }
        torch.manual_seed(31)
        one_output = self.make_model(
            4, 1,
            layer8_attention=attention,
            recurrent_memory=memory,
        )
        many_outputs = self.make_model(
            4, 5,
            layer8_attention=attention,
            recurrent_memory=memory,
        )
        many_outputs.load_state_dict(one_output.state_dict())
        for number in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
            layer = getattr(many_outputs, f"layer{number}")
            self.assertIsInstance(layer, DualStateGatedExchangeCell)
            self.assertEqual(layer.hidden_width, 4)
        self.assertIsInstance(
            many_outputs.layer8, CausalSelfAttentionLayer8
        )

        inputs = torch.randn(6, 4, 5)
        first = one_output(inputs)
        repeated = many_outputs(inputs)
        torch.testing.assert_close(first[:, 0], repeated[:, 0])
        torch.testing.assert_close(repeated, many_outputs(inputs))
        self.assertEqual(repeated.shape, (6, 5, 5))
        repeated.square().mean().backward()
        for number in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
            layer = getattr(many_outputs, f"layer{number}")
            for parameter in layer.muon_parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_dual_state_cell_uses_signed_exchange_mix(self) -> None:
        first = torch.tensor([[2.0, -1.0]])
        second = torch.tensor([[4.0, 3.0]])
        gate = torch.tensor([[0.25, 0.75]])
        actual = DualStateGatedExchangeCell.signed_mix(
            first, second, gate
        )
        torch.testing.assert_close(
            actual,
            first * gate + second * (gate - 1.0),
        )

    def test_rollout_safe_divide_has_finite_zero_denominator_gradient(
        self,
    ) -> None:
        numerator = torch.tensor([2.0, 3.0], requires_grad=True)
        denominator = torch.tensor([0.0, 4.0], requires_grad=True)
        result = _safe_divide(numerator, denominator)
        torch.testing.assert_close(result, torch.tensor([0.0, 0.75]))
        result.sum().backward()
        assert numerator.grad is not None and denominator.grad is not None
        self.assertTrue(bool(torch.isfinite(numerator.grad).all()))
        self.assertTrue(bool(torch.isfinite(denominator.grad).all()))

    def test_balanced_production59_objective_weights_and_masks_boundaries(
        self,
    ) -> None:
        derived_prediction = torch.ones(2, 2, 59)
        derived_prediction[:, :, 0] = 2.0
        derived_target = torch.zeros_like(derived_prediction)
        base_prediction = torch.full((2, 2, 17), 3.0)
        base_target = torch.zeros_like(base_prediction)
        # Only row 0, step 2 completes a minute. Masked large errors elsewhere
        # must not enter direct minute-coordinate supervision.
        base_prediction[:, :, 14:] = 100.0
        base_prediction[0, 1, 14:] = 5.0
        base_prediction.requires_grad_()
        config = {
            "weights": {
                "return": 0.5,
                "otherDerivedFeatures": 0.25,
                "primitiveCoordinates": 0.25,
            }
        }
        components = production59_balanced_objective(
            derived_prediction,
            derived_target,
            base_prediction,
            base_target,
            torch.ones(2),
            {"secondIndex": torch.tensor([57, 10])},
            config,
        )
        expected_primitive = (2 * 2 * 14 * 9 + 3 * 25) / (2 * 2 * 14 + 3)
        self.assertAlmostEqual(float(components["return"]), 4.0)
        self.assertAlmostEqual(float(components["otherDerivedFeatures"]), 1.0)
        self.assertAlmostEqual(
            float(components["primitiveCoordinates"].detach()),
            expected_primitive,
            places=5,
        )
        self.assertAlmostEqual(
            float(components["objective"].detach()),
            0.5 * 4.0 + 0.25 + 0.25 * expected_primitive,
            places=5,
        )
        components["objective"].backward()
        self.assertIsNotNone(base_prediction.grad)
        assert base_prediction.grad is not None
        self.assertGreater(
            float(base_prediction.grad[0, 1, 14:].abs().sum()), 0.0
        )
        self.assertEqual(
            float(base_prediction.grad[0, 0, 14:].abs().sum()), 0.0
        )
        self.assertEqual(
            float(base_prediction.grad[1, :, 14:].abs().sum()), 0.0
        )

    def test_layer8_knot_basis_is_causal_and_penalizes_gram(self) -> None:
        function = {
            "type": "causal-dog-knot-basis-gram-loss-v1",
            "pointWidth": 12,
            "knotWidth": 12,
            "valueWidth": 12,
            "outputWidth": 12,
            "normalizationEpsilon": 1e-6,
            "kernel": {
                "type": "normalized-distance-dog-v1",
                "bandwidth": math.sqrt(2.0),
            },
            "orthogonalizationLoss": {
                "type": "gram-identity-mean-square-v1",
                "gramEstimator": "current-batch-causal-prefix-v1",
                "weight": 1.0,
            },
        }
        torch.manual_seed(29)
        one_step = self.make_model(
            2, 1, layer8_function_approximator=function
        )
        two_steps = self.make_model(
            2, 2, layer8_function_approximator=function
        )
        two_steps.load_state_dict(one_step.state_dict())
        self.assertIsInstance(
            two_steps.layer8, CausalKnotBasisGramLossLayer8
        )

        inputs = torch.randn(512, 2, 5)
        first = one_step.trace(inputs)
        second = two_steps.trace(inputs)
        torch.testing.assert_close(
            first.expected_feature_embeddings[0],
            second.expected_feature_embeddings[0],
        )
        sequence = torch.stack(
            second.next_feature_distribution_states, dim=1
        )
        all_weights, gram_loss = \
            two_steps.layer8.basis_weights_and_gram_identity_loss(sequence)
        weights = all_weights[-1]
        points = torch.nn.functional.normalize(
            two_steps.layer8.query(sequence), dim=-1, eps=1e-6
        )
        knots = torch.nn.functional.normalize(
            two_steps.layer8.key(sequence), dim=-1, eps=1e-6
        )
        torch.testing.assert_close(
            weights,
            two_steps.layer8.kernel(points[:, -1], knots),
        )
        gram = weights.transpose(0, 1) @ weights / weights.shape[0]
        expected_loss = (
            gram - torch.eye(gram.shape[0])
        ).square().mean()
        torch.testing.assert_close(gram_loss, expected_loss)
        prediction, model_gram_loss = \
            two_steps.forward_with_auxiliary_loss(inputs)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertTrue(bool(torch.isfinite(model_gram_loss)))
        (prediction.square().mean() + model_gram_loss).backward()
        for parameter in two_steps.layer8.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_knot_basis_kernel_has_local_negative_lobe(self) -> None:
        layer = CausalKnotBasisGramLossLayer8(
            2, 2, 2, 2, 2,
            kernel_bandwidth=math.sqrt(2.0),
            normalization_epsilon=1e-6,
        )
        point = torch.tensor([[1.0, 0.0]])
        knots = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
        basis = layer.kernel(point, knots)
        self.assertAlmostEqual(float(basis[0, 0]), 1.0, places=6)
        self.assertLess(float(basis[0, 1]), 0.0)

    def test_low_rank_gnglu_uses_requested_rank_up_to_matrix_ceiling(self) -> None:
        model = self.make_model(2, 2, linear_rank=4)
        self.assertTrue(all(
            isinstance(getattr(model, f"layer{index}"), LowRankTensorPathGluBlock)
            for index in range(1, 12)
        ))
        self.assertEqual(model.layer1.projection.requested_rank, 4)
        self.assertEqual(model.layer1.projection.rank, 4)
        self.assertEqual(model.layer2.output.rank, 4)

        inputs = torch.randn(7, 2, 5)
        targets = torch.randn(7, 2, 5)
        prediction = model(inputs)
        loss = weighted_standardized_mse(
            prediction,
            model.standardized_targets(targets),
            torch.ones(7),
        )
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        for parameter in model.muon_parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_low_rank_caps_narrow_io_without_redundant_factors(self) -> None:
        model = self.make_model(1, 1, linear_rank=256)
        self.assertEqual(model.layer1.projection.requested_rank, 256)
        self.assertEqual(model.layer1.projection.rank, 5)
        self.assertEqual(model.layer2.output.requested_rank, 256)
        self.assertEqual(model.layer2.output.rank, 5)

    def test_reuses_numbered_layers_across_all_io_steps(self) -> None:
        model = self.make_model(3, 4)
        calls = {index: 0 for index in range(1, 12)}
        handles = []
        for index in calls:
            def hook(_module, _inputs, _output, *, index=index):
                calls[index] += 1
            handles.append(getattr(model, f"layer{index}").register_forward_hook(hook))
        try:
            trace = model.trace(torch.randn(2, 3, 5))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(trace.outputs.shape, (2, 4, 5))
        self.assertEqual(len(trace.feature_embeddings), 3)
        self.assertEqual(len(trace.market_states), 6)
        self.assertEqual(len(trace.next_feature_distribution_states), 4)
        self.assertEqual(calls, {
            1: 3,
            2: 4,
            3: 3,
            4: 6,
            5: 2,
            6: 2,
            7: 4,
            8: 4,
            9: 3,
            10: 3,
            11: 3,
        })

    def test_one_input_one_output_active_path_has_exact_gradient(self) -> None:
        torch.manual_seed(11)
        model = self.make_model(1, 1)
        inputs = torch.randn(7, 1, 5)
        targets = torch.randn(7, 1, 5)
        prediction = model(inputs)
        loss = weighted_standardized_mse(
            prediction,
            model.standardized_targets(targets),
            torch.ones(7),
        )
        loss.backward()
        self.assertEqual(prediction.shape, targets.shape)
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        for block in (
            model.layer1,
            model.initial_market,
            model.initial_prefix,
            model.layer3,
            model.layer4,
            model.layer7,
            model.layer8,
            model.layer2,
        ):
            gradient = block.output.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.norm()), 0.0)

    def test_optimizer_routing_covers_every_trainable_parameter_once(self) -> None:
        model = self.make_model(1, 1)
        muon, adamw = optimizer_parameter_groups(model)
        routed = {id(value) for value in (*muon, *adamw)}
        trainable = {id(value) for value in model.parameters() if value.requires_grad}
        self.assertEqual(routed, trainable)
        self.assertTrue(muon)
        self.assertTrue(adamw)

    def test_metrics_headline_raw_return_and_retain_feature_state(self) -> None:
        metrics = FeatureMetricAccumulator(
            torch.tensor([0.0, 0.0]), torch.tensor([2.0, 4.0])
        )
        metrics.add(
            torch.tensor([[[1.0, 1.0]], [[-1.0, 3.0]]]),
            torch.tensor([[[2.0, -1.0]], [[-2.0, 1.0]]]),
            torch.ones(2),
        )
        result = metrics.result()
        headline = result["nextReturn"]
        self.assertEqual(headline["evaluationScope"], COMPARABLE_EVALUATION_SCOPE)
        self.assertEqual(headline["examples"], 2)
        self.assertAlmostEqual(headline["normalizedMse"], 0.25)
        self.assertAlmostEqual(headline["mse"], 1.0)
        self.assertAlmostEqual(headline["zeroBaselineMse"], 4.0)
        self.assertAlmostEqual(headline["mseSkillVsZero"], 0.75)
        self.assertAlmostEqual(headline["directionAccuracy"], 1.0)
        self.assertAlmostEqual(headline["correlation"], 1.0)
        deciles = headline["directionByAbsoluteTargetDecile"]
        self.assertEqual(len(deciles), 2)
        self.assertEqual(sum(row["examples"] for row in deciles), 2)
        populated = [row for row in deciles if row["examples"]]
        self.assertTrue(populated)
        self.assertTrue(all(row["directionAccuracy"] == 1.0 for row in populated))
        self.assertTrue(all(
            row["magnitudeWeightedDirectionAccuracy"] == 1.0
            for row in populated
        ))
        self.assertTrue(all(row["signedReturnCapture"] == 1.0 for row in populated))
        self.assertEqual(result["featureState"]["featureCount"], 2)

    def test_sequence_metrics_headline_step_one_and_report_every_step(self) -> None:
        metrics = FeatureSequenceMetricAccumulator(
            torch.tensor([0.0, 0.0]),
            torch.tensor([2.0, 4.0]),
            output_steps=2,
        )
        metrics.add(
            torch.tensor([
                [[1.0, 1.0], [0.0, 0.0]],
                [[-1.0, 3.0], [0.0, 0.0]],
            ]),
            torch.tensor([
                [[2.0, -1.0], [4.0, 0.0]],
                [[-2.0, 1.0], [-4.0, 0.0]],
            ]),
            torch.ones(2),
        )
        result = metrics.result()
        self.assertEqual(len(result["perStepNextReturn"]), 2)
        self.assertAlmostEqual(result["nextReturn"]["normalizedMse"], 0.25)
        self.assertAlmostEqual(
            result["perStepNextReturn"][1]["normalizedMse"], 4.0
        )
        self.assertAlmostEqual(result["returnPath"]["normalizedMse"], 2.125)
        self.assertEqual(len(result["perStepFeatureState"]), 2)


class StructuredFeatureSequenceDatasetTest(unittest.TestCase):
    def test_sparse_episode_sampling_is_deterministic_and_sequential(self) -> None:
        origins, episodes = sparse_episode_origins(
            timeline_rows=100_001,
            example_count=25_000,
            target_episode_seconds=3_600,
            seed=19,
        )
        repeated, _ = sparse_episode_origins(
            timeline_rows=100_001,
            example_count=25_000,
            target_episode_seconds=3_600,
            seed=19,
        )
        np.testing.assert_array_equal(origins, repeated)
        self.assertEqual(origins.size, 25_000)
        self.assertEqual(len(episodes), 7)
        self.assertEqual(sum(episode_lengths(25_000, 3_600)), 25_000)
        self.assertLessEqual(
            max(item["examples"] for item in episodes)
            - min(item["examples"] for item in episodes),
            1,
        )
        for episode in episodes:
            selected = origins[
                (origins >= episode["startOrigin"])
                & (origins < episode["stopOriginExclusive"])
            ]
            self.assertEqual(selected.size, episode["examples"])
            self.assertTrue(bool(np.all(np.diff(selected) == 1)))

    def test_targets_are_complete_feature_rows_after_each_origin(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            files: dict[str, dict[str, str]] = {}
            timeline_rows: dict[str, int] = {}
            counts: dict[str, int] = {}
            expected_timeline = None
            for split_index, split in enumerate(("train", "validation", "test")):
                timeline = (
                    np.arange(30, dtype=np.float32).reshape(10, 3)
                    + split_index * 100
                )
                origins = np.asarray([2, 5], dtype="<i4")
                timeline_file = f"{split}.timeline-features.f32"
                origins_file = f"{split}.origins.i32"
                timeline.astype("<f4").tofile(root / timeline_file)
                origins.tofile(root / origins_file)
                files[split] = {
                    "timelineFeatures": timeline_file,
                    "origins": origins_file,
                }
                timeline_rows[split] = 10
                counts[split] = 2
                if split == "train":
                    expected_timeline = timeline
            (root / "manifest.json").write_text(json.dumps({
                "storageLayout": "temporal-channel-timeline-v1",
                "temporalChannelCount": 3,
                "examplesBySplit": counts,
                "timelineRowsBySplit": timeline_rows,
                "files": files,
            }), encoding="utf-8")
            dataset = StructuredFeatureSequenceDataset(
                root, input_steps=2, output_steps=2, train_examples=2
            )
            inputs, targets = dataset._examples(
                "train", np.asarray([0], dtype=np.int64)
            )
            assert expected_timeline is not None
            np.testing.assert_array_equal(inputs[0], expected_timeline[[1, 2]])
            np.testing.assert_array_equal(targets[0], expected_timeline[[3, 4]])
            dataset.close()


if __name__ == "__main__":
    unittest.main()
