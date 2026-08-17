"""Leakage-safe calibration of rolling probabilistic candle forecasts."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from scipy import special

from evaluate_parametric_next_day_process import ensemble_metrics
from evaluate_rolling_refit_next_day_process import (
    actual_archive_key,
    ensemble_archive_key,
)


DEFAULT_CALIBRATION_DAYS = 183
DEFAULT_OUTPUT_MEMBERS = 16
CANDIDATE_METHODS = (
    "raw",
    "residual",
    "standardizedResidual",
    "affineResidual",
    "affineStandardizedResidual",
)
LOG_FEATURES = {
    "oneSecondRealizedVarianceBpsSquared",
    "minutePathRangeBps",
    "minuteReturnQuadraticVariationBpsSquared",
    "maximumAbsoluteMinuteReturnBps",
}


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_report = read_json(repo / args.report)
    archive_path = repo / args.forecasts
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadataJson"].item()))
        report = calibrate_archive(
            archive,
            metadata=metadata,
            source_report=source_report,
            source_report_location=args.report,
            archive_location=args.forecasts,
            calibration_days=args.calibration_days,
            online_calibration_days=args.online_calibration_days,
            output_members=args.output_members,
        )
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(compact_summary(report), indent=2))


def calibrate_archive(
    archive,
    *,
    metadata: dict,
    source_report: dict,
    source_report_location: str,
    archive_location: str,
    calibration_days: int,
    online_calibration_days: int,
    output_members: int,
) -> dict:
    observations = int(metadata["targetDays"])
    if not 40 <= calibration_days <= observations - 40:
        raise ValueError("calibration split must leave at least 40 observations on each side")
    if not 30 <= online_calibration_days <= calibration_days:
        raise ValueError("online calibration history must be between 30 and calibration-days")
    selection_fit_days = calibration_days // 2
    selection_start = selection_fit_days
    windows = tuple(int(value) for value in metadata["windows"])
    horizons = tuple(metadata["horizons"])
    features = tuple(metadata["features"])

    selection = {}
    selected_methods = {}
    selected_windows = {}
    for horizon in horizons:
        selection[horizon] = {}
        selected_methods[horizon] = {}
        selected_windows[horizon] = {}
        for feature in features:
            method_scores = {}
            method_details = {}
            for method in CANDIDATE_METHODS:
                scores = []
                details = {}
                for window in windows:
                    actual = archive[actual_archive_key(horizon, feature)]
                    ensemble = archive[ensemble_archive_key(window, horizon, feature)]
                    validation_actual = actual[selection_start:calibration_days]
                    if method == "raw":
                        prediction = ensemble[selection_start:calibration_days]
                    else:
                        model = fit_calibrator(
                            feature,
                            actual[:selection_fit_days],
                            ensemble[:selection_fit_days],
                            method,
                        )
                        prediction, _ = calibrated_ensemble(
                            feature,
                            ensemble[selection_start:calibration_days],
                            model,
                            output_members,
                        )
                    metrics = ensemble_metrics(validation_actual, prediction)
                    score = calibration_selection_score(metrics)
                    scores.append(score)
                    details[str(window)] = {
                        "selectionScore": score,
                        "normalizedCrps": metrics["normalizedCrpsByActualStd"],
                        "coverageMeanAbsoluteError": calibration_assessment(metrics)[
                            "coverageMeanAbsoluteError"
                        ],
                    }
                method_scores[method] = float(np.mean(scores))
                method_details[method] = details
            chosen = min(method_scores, key=method_scores.get)
            chosen_window = min(
                windows,
                key=lambda window: method_details[chosen][str(window)][
                    "selectionScore"
                ],
            )
            selected_methods[horizon][feature] = chosen
            selected_windows[horizon][feature] = chosen_window
            selection[horizon][feature] = {
                "chosenMethod": chosen,
                "chosenHistoryWindow": chosen_window,
                "meanScoreAcrossWindows": method_scores,
                "windowDetails": method_details,
            }

    windows_report = {}
    for window in windows:
        windows_report[str(window)] = {
            "historyDays": window,
            "horizons": {},
        }
        for horizon in horizons:
            feature_report = {}
            for feature in features:
                actual = archive[actual_archive_key(horizon, feature)]
                ensemble = archive[ensemble_archive_key(window, horizon, feature)]
                raw_metrics = ensemble_metrics(
                    actual[calibration_days:],
                    ensemble[calibration_days:],
                )
                method = selected_methods[horizon][feature]
                if method == "raw":
                    calibrated = ensemble[calibration_days:]
                    parameters = {
                        "method": "raw",
                        "domain": feature_domain(feature),
                    }
                else:
                    model = fit_calibrator(
                        feature,
                        actual[:calibration_days],
                        ensemble[:calibration_days],
                        method,
                    )
                    calibrated, parameters = calibrated_ensemble(
                        feature,
                        ensemble[calibration_days:],
                        model,
                        output_members,
                    )
                calibrated_metrics = ensemble_metrics(
                    actual[calibration_days:],
                    calibrated,
                )
                raw_assessment = calibration_assessment(raw_metrics)
                calibrated_assessment = calibration_assessment(calibrated_metrics)
                feature_report[feature] = {
                    "method": method,
                    "rawTest": raw_metrics,
                    "calibratedTest": calibrated_metrics,
                    "rawAssessment": raw_assessment,
                    "calibratedAssessment": calibrated_assessment,
                    "comparison": {
                        "crpsSkill": float(
                            1.0
                            - calibrated_metrics["meanCrps"]
                            / raw_metrics["meanCrps"]
                        ),
                        "coverageMaeReduction": float(
                            raw_assessment["coverageMeanAbsoluteError"]
                            - calibrated_assessment["coverageMeanAbsoluteError"]
                        ),
                        "maximumCoverageErrorReduction": float(
                            raw_assessment["maximumAbsoluteCoverageError"]
                            - calibrated_assessment["maximumAbsoluteCoverageError"]
                        ),
                    },
                    "parameters": parameters,
                }
            windows_report[str(window)]["horizons"][horizon] = {
                "features": feature_report,
            }

    test_start = add_days(metadata["testStart"], calibration_days)
    frozen_selection_test = {
        horizon: {
            feature: frozen_selection_result(
                windows_report,
                selected_windows[horizon][feature],
                horizon,
                feature,
            )
            for feature in features
        }
        for horizon in horizons
    }
    online_selection_test = {}
    for horizon in horizons:
        online_selection_test[horizon] = {}
        for feature in features:
            window = selected_windows[horizon][feature]
            method = selected_methods[horizon][feature]
            actual = archive[actual_archive_key(horizon, feature)]
            ensemble = archive[ensemble_archive_key(window, horizon, feature)]
            online_ensemble = walk_forward_calibrated_ensemble(
                feature,
                actual,
                ensemble,
                start=calibration_days,
                method=method,
                history_days=online_calibration_days,
                output_members=output_members,
            )
            online_metrics = ensemble_metrics(
                actual[calibration_days:],
                online_ensemble,
            )
            raw_values = frozen_selection_test[horizon][feature]
            online_assessment = calibration_assessment(online_metrics)
            online_selection_test[horizon][feature] = {
                "historyWindowDays": window,
                "method": method,
                "calibrationHistoryDays": online_calibration_days,
                "rawTest": raw_values["rawTest"],
                "onlineCalibratedTest": online_metrics,
                "rawAssessment": raw_values["rawAssessment"],
                "onlineCalibratedAssessment": online_assessment,
                "comparison": metric_comparison(
                    raw_values["rawTest"],
                    online_metrics,
                ),
            }
    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": source_report["symbol"],
        "design": {
            "sourceForecastReport": source_report_location,
            "forecastArchive": archive_location,
            "forecastOriginStart": metadata["testStart"],
            "forecastOriginEndExclusive": metadata["testEndExclusive"],
            "calibrationStart": metadata["testStart"],
            "calibrationEndExclusive": test_start,
            "calibrationDays": calibration_days,
            "innerMethodFitDays": selection_fit_days,
            "innerMethodSelectionDays": calibration_days - selection_fit_days,
            "untouchedTestStart": test_start,
            "untouchedTestEndExclusive": metadata["testEndExclusive"],
            "untouchedTestDays": observations - calibration_days,
            "outputEnsembleMembers": output_members,
            "onlineCalibrationHistoryDays": online_calibration_days,
            "futureDataUsedAtForecastTime": False,
            "testOutcomesUsedForCalibration": False,
            "methodSelection": (
                "Candidate correction families are fitted on the first half of the "
                "calibration segment and selected on its second half. The selected family "
                "is refitted on the complete calibration segment, frozen, and scored only "
                "on the later untouched test segment."
            ),
            "selectionObjective": (
                "normalized CRPS plus mean absolute central-interval coverage error, "
                "plus half the PIT Kolmogorov-Smirnov statistic, averaged across "
                "history windows"
            ),
            "calibrationMeaning": (
                "This calibrates distributions of path-level statistics. It does not yet "
                "force one jointly generated candle path to have every calibrated feature."
            ),
        },
        "selection": selection,
        "windows": windows_report,
        "frozenSelectionTest": frozen_selection_test,
        "onlineSelectionTest": online_selection_test,
    }
    report["ranking"] = calibrated_ranking(report, horizons, features)
    report["aggregateAllWindows"] = aggregate_improvements(
        report,
        horizons,
        features,
    )
    report["aggregateFrozenSelection"] = aggregate_frozen_improvements(
        frozen_selection_test,
        horizons,
        features,
    )
    report["aggregateOnlineSelection"] = aggregate_online_improvements(
        online_selection_test,
        horizons,
        features,
    )
    return report


def fit_calibrator(
    feature: str,
    actual: np.ndarray,
    ensemble: np.ndarray,
    method: str,
) -> dict:
    transformed_actual = forward_transform(feature, actual)
    transformed_ensemble = forward_transform(feature, ensemble)
    center, spread = ensemble_center_spread(transformed_ensemble)
    spread_floor = max(float(np.median(spread)) * 0.1, 1e-6)
    spread = np.maximum(spread, spread_floor)
    affine = method.startswith("affine")
    standardized = "standardized" in method.lower()
    if affine:
        centered_x = center - np.mean(center)
        centered_y = transformed_actual - np.mean(transformed_actual)
        denominator = float(np.sum(centered_x * centered_x))
        raw_slope = (
            float(np.sum(centered_x * centered_y) / denominator)
            if denominator > 1e-12
            else 1.0
        )
        prior_strength = 20.0
        slope = float(np.clip(
            (actual.size * raw_slope + prior_strength) / (actual.size + prior_strength),
            0.0,
            2.5,
        ))
        intercept = float(np.mean(transformed_actual) - slope * np.mean(center))
    else:
        slope = 1.0
        intercept = 0.0
    location = intercept + slope * center
    residuals = transformed_actual - location
    if standardized:
        residuals = residuals / spread
    return {
        "method": method,
        "domain": feature_domain(feature),
        "intercept": intercept,
        "slope": slope,
        "spreadFloor": spread_floor,
        "standardized": standardized,
        "residualSamples": residuals,
    }


def calibrated_ensemble(
    feature: str,
    ensemble: np.ndarray,
    model: dict,
    output_members: int,
) -> tuple[np.ndarray, dict]:
    transformed = forward_transform(feature, ensemble)
    center, spread = ensemble_center_spread(transformed)
    spread = np.maximum(spread, model["spreadFloor"])
    location = model["intercept"] + model["slope"] * center
    probabilities = (np.arange(output_members, dtype=np.float64) + 0.5) / output_members
    residual_quantiles = np.quantile(model["residualSamples"], probabilities)
    multiplier = spread if model["standardized"] else np.ones_like(spread)
    calibrated_transformed = (
        location[:, None] + multiplier[:, None] * residual_quantiles[None, :]
    )
    calibrated = inverse_transform(feature, calibrated_transformed)
    parameters = {
        "method": model["method"],
        "domain": model["domain"],
        "intercept": model["intercept"],
        "slope": model["slope"],
        "spreadFloor": model["spreadFloor"],
        "standardizedResiduals": model["standardized"],
        "residualQuantileProbabilities": probabilities.tolist(),
        "residualQuantiles": residual_quantiles.tolist(),
    }
    return calibrated, parameters


def walk_forward_calibrated_ensemble(
    feature: str,
    actual: np.ndarray,
    ensemble: np.ndarray,
    *,
    start: int,
    method: str,
    history_days: int,
    output_members: int,
) -> np.ndarray:
    if method == "raw":
        return ensemble[start:].copy()
    result = np.empty((actual.size - start, output_members), dtype=np.float64)
    for output_index, origin in enumerate(range(start, actual.size)):
        history_start = max(0, origin - history_days)
        model = fit_calibrator(
            feature,
            actual[history_start:origin],
            ensemble[history_start:origin],
            method,
        )
        prediction, _ = calibrated_ensemble(
            feature,
            ensemble[origin:origin + 1],
            model,
            output_members,
        )
        result[output_index] = prediction[0]
    return result


def ensemble_center_spread(ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(ensemble, axis=1)
    spread = np.std(ensemble, axis=1, ddof=1)
    return center, np.maximum(spread, 1e-12)


def feature_domain(feature: str) -> str:
    if feature in LOG_FEATURES:
        return "log"
    if feature == "activeSecondFraction":
        return "logit"
    if feature == "minuteLag1ReturnCorrelation":
        return "fisherZ"
    return "identity"


def forward_transform(feature: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    domain = feature_domain(feature)
    if domain == "log":
        return np.log(np.maximum(values, 1e-12))
    if domain == "logit":
        return special.logit(np.clip(values, 1e-6, 1.0 - 1e-6))
    if domain == "fisherZ":
        return np.arctanh(np.clip(values, -1.0 + 1e-6, 1.0 - 1e-6))
    return values.copy()


def inverse_transform(feature: str, values: np.ndarray) -> np.ndarray:
    domain = feature_domain(feature)
    if domain == "log":
        return np.exp(np.clip(values, -40.0, 40.0))
    if domain == "logit":
        return special.expit(values)
    if domain == "fisherZ":
        return np.tanh(values)
    return values.copy()


def calibration_assessment(metrics: dict) -> dict:
    errors = [
        abs(values["coverageError"])
        for values in metrics["centralIntervals"].values()
    ]
    return {
        "pitUniformAtFivePercent": metrics["pit"]["ksPValue"] >= 0.05,
        "coverageMeanAbsoluteError": float(np.mean(errors)),
        "maximumAbsoluteCoverageError": float(np.max(errors)),
        "allCentralCoverageErrorsWithinFivePoints": max(errors) <= 0.05,
    }


def calibration_selection_score(metrics: dict) -> float:
    assessment = calibration_assessment(metrics)
    return float(
        metrics["normalizedCrpsByActualStd"]
        + assessment["coverageMeanAbsoluteError"]
        + 0.5 * metrics["pit"]["ksStatistic"]
    )


def metric_comparison(raw_metrics: dict, calibrated_metrics: dict) -> dict:
    raw_assessment = calibration_assessment(raw_metrics)
    calibrated_assessment = calibration_assessment(calibrated_metrics)
    return {
        "crpsSkill": float(
            1.0 - calibrated_metrics["meanCrps"] / raw_metrics["meanCrps"]
        ),
        "coverageMaeReduction": float(
            raw_assessment["coverageMeanAbsoluteError"]
            - calibrated_assessment["coverageMeanAbsoluteError"]
        ),
        "maximumCoverageErrorReduction": float(
            raw_assessment["maximumAbsoluteCoverageError"]
            - calibrated_assessment["maximumAbsoluteCoverageError"]
        ),
    }


def calibrated_ranking(report: dict, horizons: tuple[str, ...], features: tuple[str, ...]) -> dict:
    return {
        horizon: {
            "lowestCalibratedTestCrpsByFeature": {
                feature: min(
                    report["windows"],
                    key=lambda window: report["windows"][window]["horizons"][horizon][
                        "features"
                    ][feature]["calibratedTest"]["meanCrps"],
                )
                for feature in features
            }
        }
        for horizon in horizons
    }


def frozen_selection_result(
    windows_report: dict,
    window: int,
    horizon: str,
    feature: str,
) -> dict:
    values = windows_report[str(window)]["horizons"][horizon]["features"][feature]
    return {
        "historyWindowDays": window,
        "method": values["method"],
        "rawTest": values["rawTest"],
        "calibratedTest": values["calibratedTest"],
        "rawAssessment": values["rawAssessment"],
        "calibratedAssessment": values["calibratedAssessment"],
        "comparison": values["comparison"],
        "parameters": values["parameters"],
    }


def aggregate_improvements(report: dict, horizons: tuple[str, ...], features: tuple[str, ...]) -> dict:
    comparisons = [
        report["windows"][window]["horizons"][horizon]["features"][feature]
        for window in report["windows"]
        for horizon in horizons
        for feature in features
    ]
    return {
        "windowHorizonFeatureCases": len(comparisons),
        "positiveCrpsSkillCases": sum(
            values["comparison"]["crpsSkill"] > 0.0 for values in comparisons
        ),
        "reducedCoverageMaeCases": sum(
            values["comparison"]["coverageMaeReduction"] > 0.0
            for values in comparisons
        ),
        "rawPitPassCases": sum(
            values["rawAssessment"]["pitUniformAtFivePercent"]
            for values in comparisons
        ),
        "calibratedPitPassCases": sum(
            values["calibratedAssessment"]["pitUniformAtFivePercent"]
            for values in comparisons
        ),
        "rawAllCoverageWithinFivePointsCases": sum(
            values["rawAssessment"]["allCentralCoverageErrorsWithinFivePoints"]
            for values in comparisons
        ),
        "calibratedAllCoverageWithinFivePointsCases": sum(
            values["calibratedAssessment"]["allCentralCoverageErrorsWithinFivePoints"]
            for values in comparisons
        ),
    }


def aggregate_frozen_improvements(
    frozen: dict,
    horizons: tuple[str, ...],
    features: tuple[str, ...],
) -> dict:
    comparisons = [
        frozen[horizon][feature]
        for horizon in horizons
        for feature in features
    ]
    return {
        "horizonFeatureCases": len(comparisons),
        "positiveCrpsSkillCases": sum(
            values["comparison"]["crpsSkill"] > 0.0 for values in comparisons
        ),
        "reducedCoverageMaeCases": sum(
            values["comparison"]["coverageMaeReduction"] > 0.0
            for values in comparisons
        ),
        "rawPitPassCases": sum(
            values["rawAssessment"]["pitUniformAtFivePercent"]
            for values in comparisons
        ),
        "calibratedPitPassCases": sum(
            values["calibratedAssessment"]["pitUniformAtFivePercent"]
            for values in comparisons
        ),
        "rawAllCoverageWithinFivePointsCases": sum(
            values["rawAssessment"]["allCentralCoverageErrorsWithinFivePoints"]
            for values in comparisons
        ),
        "calibratedAllCoverageWithinFivePointsCases": sum(
            values["calibratedAssessment"]["allCentralCoverageErrorsWithinFivePoints"]
            for values in comparisons
        ),
    }


def aggregate_online_improvements(
    online: dict,
    horizons: tuple[str, ...],
    features: tuple[str, ...],
) -> dict:
    comparisons = [
        online[horizon][feature]
        for horizon in horizons
        for feature in features
    ]
    return {
        "horizonFeatureCases": len(comparisons),
        "positiveCrpsSkillCases": sum(
            values["comparison"]["crpsSkill"] > 0.0 for values in comparisons
        ),
        "reducedCoverageMaeCases": sum(
            values["comparison"]["coverageMaeReduction"] > 0.0
            for values in comparisons
        ),
        "rawPitPassCases": sum(
            values["rawAssessment"]["pitUniformAtFivePercent"]
            for values in comparisons
        ),
        "onlineCalibratedPitPassCases": sum(
            values["onlineCalibratedAssessment"]["pitUniformAtFivePercent"]
            for values in comparisons
        ),
        "rawAllCoverageWithinFivePointsCases": sum(
            values["rawAssessment"]["allCentralCoverageErrorsWithinFivePoints"]
            for values in comparisons
        ),
        "onlineCalibratedAllCoverageWithinFivePointsCases": sum(
            values["onlineCalibratedAssessment"][
                "allCentralCoverageErrorsWithinFivePoints"
            ]
            for values in comparisons
        ),
    }


def compact_summary(report: dict) -> dict:
    key_features = (
        "periodReturnBps",
        "oneSecondRealizedVarianceBpsSquared",
        "activeSecondFraction",
        "minutePathRangeBps",
        "maximumAbsoluteMinuteReturnBps",
    )
    horizons = {}
    for horizon, feature_values in report["frozenSelectionTest"].items():
        horizons[horizon] = {}
        for feature in key_features:
            values = feature_values[feature]
            horizons[horizon][feature] = {
                "window": values["historyWindowDays"],
                "method": values["method"],
                "rawCrps": values["rawTest"]["meanCrps"],
                "calibratedCrps": values["calibratedTest"]["meanCrps"],
                "crpsSkill": values["comparison"]["crpsSkill"],
                "rawMaxCoverageError": values["rawAssessment"][
                    "maximumAbsoluteCoverageError"
                ],
                "calibratedMaxCoverageError": values["calibratedAssessment"][
                    "maximumAbsoluteCoverageError"
                ],
            }
    return {
        "design": report["design"],
        "aggregateFrozenSelection": report["aggregateFrozenSelection"],
        "aggregateOnlineSelection": report["aggregateOnlineSelection"],
        "aggregateAllWindows": report["aggregateAllWindows"],
        "frozenSelectionByHorizon": horizons,
    }


def add_days(value: str, days: int) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed + timedelta(days=days)).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="data/benchmarks/rolling-refit-next-day-process.json",
    )
    parser.add_argument(
        "--forecasts",
        default="data/benchmarks/rolling-refit-next-day-process-forecasts.npz",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/calibrated-rolling-forecasts.json",
    )
    parser.add_argument("--calibration-days", type=int, default=DEFAULT_CALIBRATION_DAYS)
    parser.add_argument("--online-calibration-days", type=int, default=90)
    parser.add_argument("--output-members", type=int, default=DEFAULT_OUTPUT_MEMBERS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
