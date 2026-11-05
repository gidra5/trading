"""Score the retained H1 hurdle law's actual expected LOG return, without fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, roots_genlaguerre

from online_hurdle_calibration import beta_log_odds_fisher


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def log_means(packed: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gamma quadrature for E[log(1+U)], Beta identity for E[log(1-D)]."""
    if not np.all(packed[:,4:] == packed[0,4:]):
        raise ValueError("this audit expects the retained constant per-horizon shapes")
    alpha_up,alpha_down = packed[0,4:]
    nodes,weights = roots_genlaguerre(order,float(alpha_up)-1)
    weights = weights/weights.sum()
    np.testing.assert_allclose(weights@nodes,alpha_up,rtol=1e-12,atol=1e-14)
    mean_up,mean_down = packed[:,2],packed[:,3]
    expected_up = np.log1p((mean_up/alpha_up)[:,None]*nodes)@weights
    beta = alpha_down*(1-mean_down)/mean_down
    # beta_log_odds_fisher returns beta*(psi(beta+alpha)-psi(beta)),
    # using its independently checked large-beta expansion when needed.
    expected_down = np.array([-beta_log_odds_fisher(float(alpha_down),float(b))[0]/b for b in beta])
    if np.any(expected_up < 0) or np.any(expected_down >= 0):
        raise ValueError("incorrect directional log-mean signs")
    if np.any(expected_up > np.log1p(mean_up)+1e-14) or np.any(expected_down > np.log1p(-mean_down)+1e-14):
        raise ValueError("expected log means violate Jensen's inequality")
    return expected_up,expected_down


def score(y: np.ndarray, conditional: np.ndarray, deployed: np.ndarray) -> dict:
    active = y != 0
    def measured(predicted,mask):
        mse = float(np.square(predicted[mask]-y[mask]).mean())
        return {"mseBpsSquared":mse*1e8,"mseSkillVsZero":1-mse/float(np.square(y[mask]).mean())}
    return {"allTick":measured(deployed,np.ones(len(y),dtype=bool)),
        "nonzeroDeployed":measured(deployed,active),"nonzeroConditional":measured(conditional,active)}


def main() -> None:
    root = Path("data/training/runs/stochastic-crra-online-calibration-v1")
    selection_file,result_file = root/"selection.json",root/"result.json"
    selected,result = json.loads(selection_file.read_text()),json.loads(result_file.read_text())
    if selected["horizons"][0]["validationPreferredMode"] != "full":
        raise ValueError("retained H1 calibration changed")
    row = result["test"][0]["models"]["full"]
    cache_file = root/"cache"/"test-h1.npz"
    cache_manifest = json.loads(cache_file.with_suffix(".manifest.json").read_text())
    market_file = Path(cache_manifest["marketFile"])
    parameters_file = Path(row["savedStream"]["parametersFile"])
    expected_files = [(selection_file,result["selectionSha256"]),(cache_file,cache_manifest["cacheSha256"]),
        (market_file,cache_manifest["marketSha256"]),(parameters_file,row["savedStream"]["parametersSha256"])]
    for file,digest in expected_files:
        if sha(file) != digest:
            raise ValueError(f"frozen artifact changed: {file}")
    with np.load(cache_file,allow_pickle=False) as cache, np.load(market_file,allow_pickle=False) as market, \
            np.load(parameters_file,allow_pickle=False) as forecasts:
        ix = cache["accuracyIndices"]
        origins = cache["origins"][ix]
        np.testing.assert_array_equal(forecasts["origins"],cache["origins"])
        np.testing.assert_array_equal(forecasts["accuracyIndices"],ix)
        packed = forecasts["parameters"][ix]
        offsets = origins-int(market["firstPriceOrigin"])
        prices = market["prices"]
        simple = prices[offsets+1]/prices[offsets]-1
    target = np.log1p(simple)
    p0,pup = expit(packed[:,0]),expit(packed[:,1])
    low_up,low_down = log_means(packed,64)
    high_up,high_down = log_means(packed,128)
    conditional = pup*high_up+(1-pup)*high_down
    deployed = (1-p0)*conditional
    low_conditional = pup*low_up+(1-pup)*low_down
    difference = float(np.max(np.abs(low_conditional-conditional)))
    if difference > 1e-12:
        raise ValueError("log expectation quadrature did not converge")
    conditional_simple = pup*packed[:,2]-(1-pup)*packed[:,3]
    simple_scores = score(simple,conditional_simple,(1-p0)*conditional_simple)
    for key,field in (("allTick","allTickSignedMeanMseSkillVsZero"),
                      ("nonzeroDeployed","activeOnlyDeployedSignedMeanMseSkillVsZero"),
                      ("nonzeroConditional","activeOnlyConditionalSignedMeanMseSkillVsZero")):
        np.testing.assert_allclose(simple_scores[key]["mseSkillVsZero"],row[field],rtol=1e-12,atol=1e-12)
    output = {"contract":"retained-hurdle-log-return-audit-v1","horizonSeconds":1,"split":"test",
        "selectionFile":str(selection_file),"selectionSha256":sha(selection_file),
        "parametersFile":str(parameters_file),"parametersSha256":sha(parameters_file),
        "examples":len(target),"nonzeroExamples":int((target!=0).sum()),
        "originsSha256":array_sha(origins),"simpleTargetsSha256":array_sha(simple),"logTargetsSha256":array_sha(target),
        "target":"X=log(P[t+1]/P[t]) evaluated as log1p(simple return), float64",
        "prediction":"E[X] under the unchanged calibrated Gamma/Beta hurdle law; NOT log1p(E[simple return])",
        "upIntegration":"128-point generalized Gauss-Laguerre, checked against64 points",
        "downIntegration":"psi(beta)-psi(alpha+beta), with independently checked stable large-beta expansion",
        "maximum64vs128ConditionalLogMeanErrorBps":difference*1e4,
        "logScores":score(target,conditional,deployed),"simpleScores":simple_scores,
        "changesWeightsOrCalibration":False,"testUsedForSelection":False,
        "scope":"same previously inspected test origins; nonzero conditioning is an evaluation diagnostic"}
    file = Path("data/backtests/oracle-online-calibration/hurdle-log-return-audit.json")
    file.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
