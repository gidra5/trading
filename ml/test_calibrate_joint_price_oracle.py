from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from calibrate_joint_price_oracle import (
    calibrate_artifact,
    validate_action_calibration_report,
)
from evaluate_joint_price_oracle_actions import select_action_temperature


class CalibrateJointPriceOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan_file = Path("E:/Projects/trading/plan.json").resolve()
        self.plan = {"id": "candidate"}
        self.manifest = {
            "id": "candidate",
            "output": {"actionCount": 3},
            "oracle": {"options": {"friction": 0.00175, "temperature": 0.01}},
            "training": {
                "bestEpoch": 7,
                "globalStep": 123,
                "datasetFingerprint": "dataset-sha",
            },
        }
        sweep = [temperature_item(0.5, 0.4), temperature_item(1.0, 0.3)]
        self.report = {
            "version": 3,
            "evaluatedAt": "2026-08-01T00:00:00Z",
            "plan": {"id": "candidate", "file": str(self.plan_file)},
            "checkpoint": {"kind": "best", "epoch": 7, "globalStep": 123},
            "split": {"name": "validation", "examples": 100},
            "oracle": {"actionCount": 3, "friction": 0.00175, "temperature": 0.01},
            "actionEvaluation": {
                "executionPolicy": {
                    "version": 2,
                    "maximumLeverage": 100.0,
                    "minimumConfidence": 0.05,
                    "confidenceExposurePower": 0.0,
                    "confidenceLeverageFloor": 0.75,
                },
                "rolloutScoreVersion": 2,
                "rolloutResetPolicy": "reset-only-at-true-timeline-gaps",
            },
            "temperatureSweep": sweep,
            "actionTemperatureCalibration": select_action_temperature(sweep),
            "datasetFingerprint": "dataset-sha",
        }

    def test_matching_validation_report_is_accepted(self) -> None:
        selection = validate_action_calibration_report(
            self.report,
            self.plan,
            self.plan_file,
            self.manifest,
        )
        self.assertEqual(selection["selectedLogitTemperature"], 0.5)
        self.assertEqual(selection["identityIndex"], 1)

    def test_exact_state_validation_metric_controls_calibration(self) -> None:
        report = copy.deepcopy(self.report)
        first, identity = report["temperatureSweep"]
        first["exactStateActions"] = exact_state_item(0.2)
        identity["exactStateActions"] = exact_state_item(0.8)
        report["actionTemperatureCalibration"] = select_action_temperature(
            report["temperatureSweep"]
        )
        selection = validate_action_calibration_report(
            report,
            self.plan,
            self.plan_file,
            self.manifest,
        )
        self.assertEqual(selection["selectedLogitTemperature"], 1.0)
        self.assertEqual(
            selection["objective"]["metric"],
            "exactStateActions.signedTransitionF1",
        )

    def test_test_report_is_never_accepted_for_calibration(self) -> None:
        report = copy.deepcopy(self.report)
        report["split"]["name"] = "test"
        with self.assertRaises(PermissionError):
            validate_action_calibration_report(
                report,
                self.plan,
                self.plan_file,
                self.manifest,
            )

    def test_stale_checkpoint_and_dataset_reports_are_rejected(self) -> None:
        for field, value in (("epoch", 8), ("globalStep", 124)):
            report = copy.deepcopy(self.report)
            report["checkpoint"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_action_calibration_report(
                    report,
                    self.plan,
                    self.plan_file,
                    self.manifest,
                )
        report = copy.deepcopy(self.report)
        report["datasetFingerprint"] = "different"
        with self.assertRaises(ValueError):
            validate_action_calibration_report(
                report,
                self.plan,
                self.plan_file,
                self.manifest,
            )

    def test_manual_temperature_override_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        selection = report["actionTemperatureCalibration"]
        selection["selectedIndex"] = 1
        selection["selectedLogitTemperature"] = 1.0
        selection["selectedMetrics"] = {
            "raw": report["temperatureSweep"][1]["raw"],
            "actions": report["temperatureSweep"][1]["actions"],
        }
        with self.assertRaisesRegex(ValueError, "deterministically derived"):
            validate_action_calibration_report(
                report,
                self.plan,
                self.plan_file,
                self.manifest,
            )

    @patch("calibrate_joint_price_oracle.apply_action_calibration")
    @patch("calibrate_joint_price_oracle.evaluate_plan_checkpoint")
    def test_workflow_has_no_test_split_option(
        self,
        evaluate,
        apply_calibration,
    ) -> None:
        evaluate.return_value = self.report
        apply_calibration.return_value = self.manifest
        calibrate_artifact(
            Path("plan.json"),
            requested_device="cpu",
            batch_size=16,
            logit_temperatures=(0.5, 1.0),
        )
        evaluate.assert_called_once_with(
            Path("plan.json"),
            checkpoint_kind="best",
            split="validation",
            allow_test=False,
            output=None,
            requested_device="cpu",
            batch_size=16,
            logit_temperatures=(0.5, 1.0),
        )


def temperature_item(temperature: float, f1: float) -> dict:
    return {
        "logitTemperature": temperature,
        "raw": {
            "crossEntropy": 1.0,
            "klDivergence": 0.5,
            "targetEntropy": 0.5,
            "predictedEntropy": 0.5,
        },
        "actions": {
            "signedTransitionF1": f1,
            "signedTransitionPrecision": f1,
            "signedTransitionRecall": f1,
            "exactTransitionF1": f1,
            "pathDirectionalAgreement": f1,
            "pathMeanAbsoluteError": 1.0,
            "turnoverRatio": 1.0,
        },
    }


def exact_state_item(f1: float) -> dict:
    return {
        "signedTransitionF1": f1,
        "signedTransitionPrecision": f1,
        "signedTransitionRecall": f1,
        "exactTransitionF1": f1,
        "executableTargetDirectionalAgreement": f1,
        "executableTargetMeanAbsoluteError": 1.0,
        "turnoverRelativeError": 0.1,
        "exactStateScore": 1 - f1,
    }


if __name__ == "__main__":
    unittest.main()
