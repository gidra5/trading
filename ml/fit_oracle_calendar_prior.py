"""Fit a deterministic fixed UTC calendar prior from training targets only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_causal_oracle_predictability import (
    ACTION_COUNT,
    read_target_day,
    split_and_purge,
)
from audit_oracle_calendar_predictability import (
    CALENDAR_SPECS,
    calendar_rows_for_segments,
    fit_calendar_table,
)
from oracle_calendar_prior import (
    build_calendar_prior_document,
    calendar_prior_document_bytes,
)
from trading_storage import training_storage_layout, write_shard_payload


DEFAULT_PLAN = Path(
    "ml/training-plans/oracle-calendar-prior-hour-weekday-v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = (
        arguments.plan
        if arguments.plan.is_absolute()
        else repo_root / arguments.plan
    ).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_artifact_plan(plan)
    target_root = (repo_root / plan["targetReferenceDir"]).resolve()
    target_files = sorted(target_root.glob("*.json"))
    segments = split_and_purge(target_files)
    train_files = {
        segment.target_file.resolve() for segment in segments["train"]
    }
    validation_or_test_files = {
        segment.target_file.resolve()
        for split in ("validation", "test")
        for segment in segments[split]
    }
    cache: dict[Path, np.ndarray] = {}
    opened: set[Path] = set()
    parts: list[np.ndarray] = []
    for segment in segments["train"]:
        day = read_target_day(segment.target_file, cache, opened)
        start = segment.target_row_offset
        parts.append(np.asarray(
            day[start:start + segment.count],
            dtype=np.float32,
        ))
    if opened != train_files or opened & validation_or_test_files:
        raise RuntimeError("calendar prior fitter crossed the training boundary")
    targets = np.concatenate(parts, axis=0)
    calendar = calendar_rows_for_segments(segments["train"])
    if calendar.values.shape[0] != targets.shape[0]:
        raise RuntimeError("calendar prior training rows are misaligned")
    calendar_plan = plan["calendarEstimator"]
    spec = next(
        candidate for candidate in CALENDAR_SPECS
        if candidate.name == calendar_plan["spec"]
    )
    table = fit_calendar_table(
        calendar.values,
        targets,
        spec,
        float(calendar_plan["finePriorStrength"]),
        float(calendar_plan["backoffPriorStrength"]),
    )
    manifest = []
    for target_file in sorted(train_files):
        reference = json.loads(target_file.read_text(encoding="utf-8"))
        manifest.append((
            target_file.stem,
            reference["object"]["contentHash"],
        ))
    manifest_sha256 = hashlib.sha256(json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    document = build_calendar_prior_document(
        artifact_id=plan["id"],
        table=table,
        fields=spec.fields,
        estimator={
            "spec": spec.name,
            "backoffFields": list(spec.backoff_fields),
            "finePriorStrength": float(
                calendar_plan["finePriorStrength"]
            ),
            "backoffPriorStrength": float(
                calendar_plan["backoffPriorStrength"]
            ),
        },
        oracle_target=plan["oracleTarget"],
        training_corpus={
            "targetReferenceDir": plan["targetReferenceDir"],
            "fileCount": len(train_files),
            "rowsAfterPurge": int(targets.shape[0]),
            "dateStart": min(path.stem for path in train_files),
            "dateEnd": max(path.stem for path in train_files),
            "referenceManifestSha256": manifest_sha256,
            "validationTargetReferencesOpened": 0,
            "testTargetReferencesOpened": 0,
        },
        selection_evidence=plan["selectionEvidence"],
    )
    storage = training_storage_layout(repo_root)
    artifact_plan = plan["artifact"]
    reference_file = write_shard_payload(
        storage.immutable,
        artifact_plan["namespace"],
        artifact_plan["key"],
        calendar_prior_document_bytes(document),
        sequence={"start": 0, "step": 1, "count": 1, "unit": "index"},
        layout={
            "encoding": "canonical-json-utf8-v1",
            "schema": "utc-calendar-table-float32-raw01-policy-v1",
        },
        metadata={
            "planId": plan["id"],
            "trainRows": int(targets.shape[0]),
            "calendarSpec": spec.name,
            "validationTargetReferencesOpened": 0,
            "testTargetReferencesOpened": 0,
        },
    )
    print(json.dumps({
        "artifactReference": str(reference_file.relative_to(repo_root)),
        "id": plan["id"],
        "trainRows": int(targets.shape[0]),
        "cells": int(table.shape[0]),
        "actions": ACTION_COUNT,
        "validationTargetReferencesOpened": 0,
        "testTargetReferencesOpened": 0,
    }, indent=2))


def validate_artifact_plan(plan: object) -> None:
    if not isinstance(plan, dict) \
            or plan.get("version") != 1 \
            or plan.get("testPolicy") != "sealed-never-load":
        raise ValueError("calendar prior plan contract is incompatible")
    calendar = plan.get("calendarEstimator")
    if not isinstance(calendar, dict):
        raise ValueError("calendar prior estimator is missing")
    spec = next((
        candidate for candidate in CALENDAR_SPECS
        if candidate.name == calendar.get("spec")
    ), None)
    if spec is None \
            or list(spec.fields) != calendar.get("fields") \
            or list(spec.backoff_fields) != calendar.get("backoffFields") \
            or float(calendar.get("finePriorStrength", 0)) <= 0 \
            or float(calendar.get("backoffPriorStrength", 0)) <= 0:
        raise ValueError("calendar prior estimator configuration is invalid")
    required = (
        "id",
        "targetReferenceDir",
        "artifact",
        "oracleTarget",
        "selectionEvidence",
    )
    if any(key not in plan for key in required):
        raise ValueError("calendar prior plan is incomplete")
    artifact = plan["artifact"]
    if not isinstance(artifact, dict) \
            or set(artifact) != {"namespace", "key"} \
            or not all(isinstance(value, str) and value for value in artifact.values()):
        raise ValueError("calendar prior immutable artifact path is invalid")


if __name__ == "__main__":
    main()
