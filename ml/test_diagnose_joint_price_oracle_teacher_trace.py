import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from diagnose_joint_price_oracle_teacher_trace import (
    _requested_execution_targets,
    build_exact_trace_exposure_provider,
    diagnose_teacher_trace,
)
from joint_price_oracle_actions import (
    execution_policy_native_scale,
    teacher_actions_at_current_exposures_numpy,
)
from trading_storage import write_shard_payload


class TeacherTraceDiagnosticTests(unittest.TestCase):
    def test_v2_exact_one_step_and_native_provider_expose_surrogate_drift(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(
                Path(temporary),
                version=2,
                maximum_leverage=1.0,
                current_exposures=np.asarray([0.0, 0.7, -0.4, 0.2]),
            )

            report = diagnose_teacher_trace(
                fixture["trace"],
                fixture["targetRoot"],
            )

            self.assertEqual(report["trace"]["schemaVersion"], 2)
            self.assertEqual(report["policy"]["executionScale"], 0.5)
            self.assertEqual(
                report["comparisons"]["immutableRawModalVsTrace"]["exactRate"],
                1.0,
            )
            one_step = report["comparisons"]["oneStepAtActualCurrentVsTrace"]
            self.assertEqual(one_step["conditionedActionIndex"]["exactRate"], 1.0)
            self.assertEqual(one_step["requestedTargetExposure"]["exactRate"], 1.0)
            self.assertEqual(one_step["signalEmission"]["divergenceCount"], 0)
            self.assertGreater(
                report["comparisons"]["surrogateRolloutVsTrace"]
                ["currentExposure"]["meanAbsoluteError"],
                0,
            )

            provider = build_exact_trace_exposure_provider(
                fixture["trace"],
                fixture["targetRoot"],
                expected_sha256=hashlib.sha256(
                    fixture["trace"].read_bytes()
                ).hexdigest(),
                expected_split="validation",
                expected_schema_version=2,
                expected_dates=("2026-01-01",),
                expected_row_count=4,
                expected_execution_policy={
                    "version": 2,
                    "maximumLeverage": 1,
                    "minimumConfidence": 0.05,
                    "confidenceExposurePower": 0,
                    "confidenceLeverageFloor": 0.75,
                },
            )
            values = provider("validation", SimpleNamespace(
                count=2,
                prediction_time_start=fixture["timestamps"][1],
                step_ms=60_000,
            ))
            np.testing.assert_allclose(values, np.asarray([1.4, -0.8]))
            self.assertEqual(provider.schema_version, 2)
            self.assertEqual(provider.row_count, 4)
            self.assertEqual(provider.maximum_leverage, 1)
            self.assertEqual(
                provider.source_sha256,
                hashlib.sha256(fixture["trace"].read_bytes()).hexdigest(),
            )

            with self.assertRaisesRegex(ValueError, "sha256"):
                build_exact_trace_exposure_provider(
                    fixture["trace"],
                    fixture["targetRoot"],
                    expected_sha256="0" * 64,
                )

    def test_v1_maps_legacy_fields_and_reports_raw_modal_unavailable(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(
                Path(temporary),
                version=1,
                maximum_leverage=100.0,
                current_exposures=np.asarray([0.0, 0.3, -0.2, 0.4]),
            )

            report = diagnose_teacher_trace(
                fixture["trace"],
                fixture["targetRoot"],
                legacy_maximum_leverage=100.0,
            )

            self.assertEqual(report["trace"]["schemaVersion"], 1)
            self.assertEqual(
                report["trace"]["maximumLeverageSource"],
                "caller-verified-legacy-v1",
            )
            self.assertFalse(
                report["comparisons"]["immutableRawModalVsTrace"]["available"]
            )
            one_step = report["comparisons"]["oneStepAtActualCurrentVsTrace"]
            self.assertEqual(one_step["conditionedModalExposure"]["exactRate"], 1.0)
            self.assertEqual(one_step["requestedTargetExposure"]["exactRate"], 1.0)
            self.assertEqual(one_step["signalEmission"]["divergenceCount"], 0)

    def test_test_trace_is_rejected_before_target_payload_is_read(self) -> None:
        with TemporaryDirectory() as temporary:
            trace = Path(temporary) / "test.json"
            trace.write_text(json.dumps({
                "version": 1,
                "kind": "exact-hindsight-oracle-bot-teacher-trace",
                "split": "test",
                "dates": ["2026-01-01"],
                "oracle": {
                    "valueHorizonSteps": 3_600,
                    "decisionDelaySteps": 60,
                    "holdingPeriodSteps": 60,
                },
                "decisions": [{
                    "timestamp": 1_000,
                    "currentExposure": 0,
                    "modalExposure": 0,
                    "targetExposure": 0,
                    "confidence": 1,
                    "entropy": 0,
                    "emitted": False,
                }],
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires --allow-test"):
                diagnose_teacher_trace(
                    trace,
                    Path(temporary) / "does-not-exist",
                )

    def _fixture(
        self,
        root: Path,
        *,
        version: int,
        maximum_leverage: float,
        current_exposures: np.ndarray,
    ) -> dict:
        date = "2026-01-01"
        start = 1_000
        step = 60_000
        grid = np.asarray([-2.0, 0.0, 2.0], dtype=np.float64)
        probabilities = np.asarray([
            [0.05, 0.10, 0.85],
            [0.80, 0.15, 0.05],
            [0.05, 0.20, 0.75],
            [0.70, 0.20, 0.10],
        ], dtype=np.float32)
        policy = {
            "version": 2,
            "maximumLeverage": maximum_leverage,
            "minimumConfidence": 0.05,
            "confidenceExposurePower": 0.0,
            "confidenceLeverageFloor": 0.75,
        }
        scale = execution_policy_native_scale(grid, policy)
        one_step = teacher_actions_at_current_exposures_numpy(
            probabilities,
            grid,
            current_exposures / scale,
            friction=0.00175,
            temperature=0.01,
            execution_policy=policy,
        )
        requested = _requested_execution_targets(
            one_step.target_indices,
            one_step.confidences,
            grid,
            policy,
            scale,
        )
        timestamps = start + np.arange(probabilities.shape[0]) * step
        raw_modal = grid[probabilities.argmax(axis=-1)]
        conditioned = grid[one_step.target_indices]
        decisions = []
        for index, timestamp in enumerate(timestamps):
            common = {
                "timestamp": int(timestamp),
                "currentExposure": float(current_exposures[index]),
                "targetExposure": float(requested[index]),
                "confidence": float(one_step.confidences[index]),
                "entropy": float(one_step.conditional_entropies[index]),
            }
            if version == 1:
                common.update({
                    "modalExposure": float(conditioned[index]),
                    "emitted": bool(one_step.switch_labels[index]),
                })
            else:
                common.update({
                    "rawModalExposure": float(raw_modal[index]),
                    "conditionedModalExposure": float(conditioned[index]),
                    "signalEmitted": bool(one_step.switch_labels[index]),
                })
            decisions.append(common)

        storage = root / "immutable"
        namespace = "oracle/fixture"
        contract = {
            "version": 1,
            "intervalMs": 1_000,
            "decisionIntervalMs": step,
            "options": {
                "holdingPeriodSteps": 60,
                "decisionDelaySteps": 60,
                "valueHorizonSteps": 3_600,
                "friction": 0.00175,
                "temperature": 0.01,
            },
            "usableGrid": grid.tolist(),
        }
        write_shard_payload(
            storage,
            namespace,
            date,
            probabilities.astype("<f4").tobytes(),
            sequence={
                "start": start,
                "step": step,
                "count": probabilities.shape[0],
                "unit": "unix-ms",
            },
            layout={
                "encoding": "raw-row-major",
                "dtype": "float32-le",
                "rows": probabilities.shape[0],
                "columns": probabilities.shape[1],
            },
            metadata={
                "contractHash": "fixture-contract",
                "contract": contract,
            },
        )
        oracle = {
            "valueHorizonSteps": 3_600,
            "decisionDelaySteps": 60,
            "holdingPeriodSteps": 60,
        }
        if version == 2:
            oracle["maximumLeverage"] = maximum_leverage
        trace_file = root / f"trace-v{version}.json"
        trace_file.write_text(json.dumps({
            "version": version,
            "kind": "exact-hindsight-oracle-bot-teacher-trace",
            "split": "validation",
            "dates": [date],
            "oracle": oracle,
            "decisions": decisions,
        }), encoding="utf-8")
        return {
            "trace": trace_file,
            "targetRoot": storage / "refs" / "oracle" / "fixture",
            "timestamps": timestamps,
        }


if __name__ == "__main__":
    unittest.main()
