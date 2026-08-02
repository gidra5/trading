"""Versioned deterministic artifact contract for a fixed calendar prior."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from audit_oracle_calendar_predictability import (
    ACTION_COUNT,
    FEATURE_CARDINALITY,
    FEATURE_NAMES,
    encode_cells,
)
from trading_storage import read_shard_payload


ARTIFACT_VERSION = 1
ARTIFACT_KIND = "causal-oracle-calendar-prior"
ARTIFACT_CONTRACT = "utc-calendar-table-float32-raw01-policy-v1"


@dataclass(frozen=True)
class CalendarPriorArtifact:
    metadata: dict[str, object]
    fields: tuple[str, ...]
    probabilities_by_cell: np.ndarray

    def probabilities(self, calendar_values: np.ndarray) -> np.ndarray:
        ids, cell_count = encode_cells(calendar_values, self.fields)
        if cell_count != self.probabilities_by_cell.shape[0]:
            raise RuntimeError("calendar artifact cell cardinality changed")
        return self.probabilities_by_cell[ids]


def build_calendar_prior_document(
    *,
    artifact_id: str,
    table: np.ndarray,
    fields: tuple[str, ...],
    estimator: dict[str, object],
    oracle_target: dict[str, object],
    training_corpus: dict[str, object],
    selection_evidence: dict[str, object],
) -> dict[str, object]:
    values = validated_table(table, fields)
    table_bytes = values.tobytes(order="C")
    return {
        "version": ARTIFACT_VERSION,
        "kind": ARTIFACT_KIND,
        "contract": ARTIFACT_CONTRACT,
        "id": artifact_id,
        "testPolicy": "sealed-never-load",
        "oracleTarget": oracle_target,
        "calendarEncoding": {
            "timezone": "UTC",
            "fields": list(fields),
            "fieldCardinalities": {
                field: FEATURE_CARDINALITY[field] for field in fields
            },
            "minuteTimestampPhaseMs": 999,
        },
        "estimator": estimator,
        "trainingCorpus": training_corpus,
        "selectionEvidence": selection_evidence,
        "table": {
            "dtype": "float32-le",
            "shape": list(values.shape),
            "sha256": hashlib.sha256(table_bytes).hexdigest(),
            "values": values.tolist(),
        },
    }


def write_calendar_prior_document(
    document: dict[str, object],
    output_file: Path,
) -> None:
    # Validate before replacing a durable artifact.  Canonical key ordering and
    # the absence of timestamps make identical inputs byte-for-byte identical.
    artifact_from_document(document)
    payload = calendar_prior_document_bytes(document).decode("utf-8")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f"{output_file.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, output_file)


def load_calendar_prior_artifact(path: Path) -> CalendarPriorArtifact:
    outer = json.loads(path.read_text(encoding="utf-8"))
    if outer.get("kind") == "trading-sequential-shard":
        shard, payload = read_shard_payload(path)
        if shard.reference.get("layout") != {
            "encoding": "canonical-json-utf8-v1",
            "schema": ARTIFACT_CONTRACT,
        }:
            raise ValueError("calendar prior storage layout is incompatible")
        outer = json.loads(payload.decode("utf-8"))
    return artifact_from_document(outer)


def calendar_prior_document_bytes(document: dict[str, object]) -> bytes:
    artifact_from_document(document)
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def artifact_from_document(document: object) -> CalendarPriorArtifact:
    if not isinstance(document, dict) \
            or document.get("version") != ARTIFACT_VERSION \
            or document.get("kind") != ARTIFACT_KIND \
            or document.get("contract") != ARTIFACT_CONTRACT \
            or document.get("testPolicy") != "sealed-never-load":
        raise ValueError("calendar prior artifact contract is incompatible")
    encoding = document.get("calendarEncoding")
    table_document = document.get("table")
    if not isinstance(encoding, dict) \
            or encoding.get("timezone") != "UTC" \
            or encoding.get("minuteTimestampPhaseMs") != 999 \
            or not isinstance(table_document, dict) \
            or table_document.get("dtype") != "float32-le":
        raise ValueError("calendar prior encoding is incompatible")
    raw_fields = encoding.get("fields")
    if not isinstance(raw_fields, list) \
            or not raw_fields \
            or any(field not in FEATURE_NAMES for field in raw_fields) \
            or len(set(raw_fields)) != len(raw_fields):
        raise ValueError("calendar prior fields are invalid")
    fields = tuple(raw_fields)
    expected_cardinalities = {
        field: FEATURE_CARDINALITY[field] for field in fields
    }
    if encoding.get("fieldCardinalities") != expected_cardinalities:
        raise ValueError("calendar prior field cardinalities changed")
    shape = table_document.get("shape")
    expected_cells = int(np.prod([
        FEATURE_CARDINALITY[field] for field in fields
    ], dtype=np.int64))
    if shape != [expected_cells, ACTION_COUNT]:
        raise ValueError("calendar prior table shape is invalid")
    table = validated_table(
        np.asarray(table_document.get("values"), dtype="<f4"),
        fields,
    )
    digest = hashlib.sha256(table.tobytes(order="C")).hexdigest()
    if table_document.get("sha256") != digest:
        raise ValueError("calendar prior table checksum mismatch")
    metadata = {
        key: value for key, value in document.items() if key != "table"
    }
    return CalendarPriorArtifact(
        metadata=metadata,
        fields=fields,
        probabilities_by_cell=table,
    )


def validated_table(table: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    expected_cells = int(np.prod([
        FEATURE_CARDINALITY[field] for field in fields
    ], dtype=np.int64))
    values = np.ascontiguousarray(table, dtype="<f4")
    if values.shape != (expected_cells, ACTION_COUNT) \
            or not np.isfinite(values).all() \
            or bool((values <= 0).any()) \
            or not np.allclose(
                values.sum(axis=1),
                1,
                atol=2e-6,
                rtol=2e-6,
            ):
        raise ValueError("calendar prior table probabilities are invalid")
    return values
