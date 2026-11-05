"""Native expected-log-return scoring of the surviving frozen joint law.

Evaluate the existing density on matched origins, without training or selection.
Expected log returns are integrated under the law, not obtained by logging its
expected simple return. Simple-return rescoring reproduces the previous audit.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import component_log_masses, load_timeline_split
from train_neural_sign_size_forecast import now, select_origins, sha, write_json

CONTRACT = "frozen-joint-native-log-return-audit-v1"


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def score_means(target: np.ndarray, conditional: np.ndarray, deployed: np.ndarray) -> dict:
    active = target != 0
    if not active.any() or not np.isfinite(np.stack((target, conditional, deployed))).all():
        raise ValueError("finite forecast/target values and active endpoints are required")
    conditional_mse = float(np.square(conditional[active]-target[active]).mean())
    deployed_active_mse = float(np.square(deployed[active]-target[active]).mean())
    deployed_mse = float(np.square(deployed-target).mean())
    active_baseline, baseline = float(np.square(target[active]).mean()), float(np.square(target).mean())
    return {"examples": len(target), "activeExamples": int(active.sum()),
        "activeConditionalMseBpsSquared": conditional_mse*1e8,
        "activeConditionalMseSkillVsZero": 1-conditional_mse/active_baseline,
        "activeDeployedMseBpsSquared": deployed_active_mse*1e8,
        "activeDeployedMseSkillVsZero": 1-deployed_active_mse/active_baseline,
        "allTickDeployedMseBpsSquared": deployed_mse*1e8,
        "allTickDeployedMseSkillVsZero": 1-deployed_mse/baseline,
        "activeZeroBaselineMseBpsSquared": active_baseline*1e8,
        "allTickZeroBaselineMseBpsSquared": baseline*1e8}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-audit", type=Path, default=Path("data/backtests/oracle-neural-specialists/joint-marginal-audit.json"))
    parser.add_argument("--online-root", type=Path, default=Path("data/training/runs/stochastic-crra-online-calibration-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-online-calibration/joint-log-return-audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    old = json.loads(args.joint_audit.read_text())
    previous = next(row for row in old["results"] if row["horizonSeconds"] == 1 and row["role"] == "previouslySelectedDensity")
    checkpoint = Path(previous["checkpoint"])
    if sha(checkpoint) != previous["sha256"]:
        raise ValueError("frozen joint checkpoint changed")
    manifest_file = args.online_root/"cache/test-h1.manifest.json"
    manifest = json.loads(manifest_file.read_text())
    for key, digest in (("cacheFile", "cacheSha256"), ("marketFile", "marketSha256")):
        if sha(Path(manifest[key])) != manifest[digest]:
            raise ValueError("frozen matched origins or prices changed")
    with np.load(manifest["cacheFile"], allow_pickle=False) as cached:
        origins = cached["origins"][cached["accuracyIndices"]]
        timestamps = cached["timestamps"][cached["accuracyIndices"]]
    with np.load(manifest["marketFile"], allow_pickle=False) as market:
        np.testing.assert_array_equal(origins, market["commonOrigins"])
        offsets = origins-int(market["firstPriceOrigin"])
        simple_targets = market["prices"][offsets+1]/market["prices"][offsets]-1
    log_targets = np.log1p(simple_targets)
    np.testing.assert_array_equal(log_targets == 0, simple_targets == 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_horizon_forecast(checkpoint, device)
    dataset = Path(payload["plan"]["dataset"])
    if sha(dataset/"manifest.json") != old["datasetManifestSha256"]:
        raise ValueError("joint and comparison datasets differ")
    timeline = load_timeline_split(dataset, "test")
    np.testing.assert_array_equal(origins, select_origins(timeline, 4, None))
    np.testing.assert_array_equal(timestamps, timeline.axis_start_ms+origins*1000)
    features = timeline.features(origins)
    p0_parts, mass_parts, native_parts = [], [], []
    batch = old["evaluationBatchSize"]
    with torch.inference_mode():
        for begin in range(0, len(origins), batch):
            logits = model(torch.from_numpy(features[begin:begin+batch]).to(device))
            p0_parts.append(torch.sigmoid(logits[:, 0]).cpu().numpy().astype(np.float64))
            mass_parts.append(component_log_masses(logits[:, 1:], model.areas).exp().cpu().numpy().astype(np.float64))
            native_parts.append(model.expectation_from_logits(logits).cpu().numpy().astype(np.float64))
    p0, masses, native = np.concatenate(p0_parts), np.concatenate(mass_parts), np.concatenate(native_parts)
    if not np.isfinite(masses).all() or np.any(masses < 0):
        raise ValueError("invalid frozen joint density weights")
    normalized = masses/masses.sum(axis=1, keepdims=True)
    means, coefficients = {}, {}
    for order in (192, 384, 768):
        coefficients[order] = model._component_moments("log", order)
        conditional = normalized@coefficients[order]
        means[order] = (conditional, (1-p0)*conditional)
    convergence = {}
    for low, high in ((192, 384), (384, 768)):
        convergence[f"{low}vs{high}"] = {
            "maximumConditionalMeanDifferenceBps": float(np.max(np.abs(means[low][0]-means[high][0]))*1e4),
            "maximumDeployedMeanDifferenceBps": float(np.max(np.abs(means[low][1]-means[high][1]))*1e4),
            "allTickSkillDifference": score_means(log_targets, *means[high])["allTickDeployedMseSkillVsZero"]
                                      -score_means(log_targets, *means[low])["allTickDeployedMseSkillVsZero"]}
    if convergence["384vs768"]["maximumDeployedMeanDifferenceBps"] > 1e-6:
        raise ValueError("expected-log-return quadrature did not converge within declared tolerance")
    simple_conditional = normalized@model._component_moments("excess", 384)
    simple_deployed = (1-p0)*simple_conditional
    simple = score_means(simple_targets, simple_conditional, simple_deployed)
    reference = previous["splits"]["test"]["mixtures"]["jointNativeZero"]
    expected = {"activeConditionalMseSkillVsZero": reference["activeOnly"]["conditionalActiveMeanMseSkillVsZero"],
        "activeDeployedMseSkillVsZero": reference["activeOnly"]["deployedUnconditionalMeanMseSkillVsZero"],
        "allTickDeployedMseSkillVsZero": reference["allTickSignedMeanMseSkillVsZero"]}
    for key, value in expected.items():
        np.testing.assert_allclose(simple[key], value, rtol=0, atol=2e-12)
    primary = score_means(log_targets, *means[768])
    shortcut = np.log1p(simple_deployed)
    output = {"contract": CONTRACT, "createdAt": now(), "horizonSeconds": 1, "split": "test",
        "checkpoint": str(checkpoint), "checkpointSha256": sha(checkpoint),
        "sourceAudit": str(args.joint_audit), "sourceAuditSha256": sha(args.joint_audit),
        "marketFile": manifest["marketFile"], "marketSha256": manifest["marketSha256"],
        "datasetManifestSha256": old["datasetManifestSha256"],
        "sampleHashes": {"origins": array_sha(origins), "timestamps": array_sha(timestamps),
            "simpleTargets": array_sha(simple_targets), "logTargets": array_sha(log_targets)},
        "firstTimestamp": int(timestamps[0]), "lastTimestamp": int(timestamps[-1]),
        "target": "X=log(P[t+1]/P[t]) computed as float64 log1p of exact endpoint simple return",
        "prediction": "integrate X under normalized conditional Gaussian-CDF triangular density; deployed expectation=(1-p0)*conditional expectation",
        "activeScoring": "both conditional and deployed forecasts evaluated only where realized X is nonzero; active conditional forecast is a diagnostic",
        "logReturns": primary, "simpleReturnRegression": simple, "simpleReturnPriorAuditParity": True,
        "quadrature": {"method": "Gauss-Legendre per triangular knot interval", "primaryOrder": 768,
            "convergence": convergence, "logComponentCoefficientsSha256": array_sha(coefficients[768]),
            "nativeFloat32ApiUsesOrder": 96,
            "maximumNativeApiVsFloat64PrimaryDifferenceBps": float(np.max(np.abs(native-means[768][1]))*1e4)},
        "incorrectLogOfSimpleMeanShortcut": {"maximumDifferenceFromExpectedLogBps": float(np.max(np.abs(shortcut-means[768][1]))*1e4),
            "allTickDeployedMseSkillVsZero": float(1-np.square(shortcut-log_targets).mean()/np.square(log_targets).mean())},
        "inference": {"device": str(device), "batchSize": batch, "matmulPrecision": "high"},
        "sourceTrainingObjective": payload["plan"]["trainingLoss"],
        "changesTrainingOrSelection": False, "seconds": time.monotonic()-started}
    if sha(checkpoint) != previous["sha256"]:
        raise ValueError("source checkpoint changed during audit")
    write_json(args.output, output)
    print(json.dumps({"event": "jointLogReturnAuditComplete", "output": str(args.output), "sha256": sha(args.output),
        "sampleHashes": output["sampleHashes"], "logReturns": primary, "simpleReturnRegression": simple,
        "quadrature": output["quadrature"], "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
