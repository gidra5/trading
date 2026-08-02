from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_oracle_calendar_predictability import (
    ACTION_COUNT,
    FEATURE_INDEX,
)
from fit_oracle_calendar_prior import validate_artifact_plan
from oracle_calendar_prior import (
    artifact_from_document,
    build_calendar_prior_document,
    calendar_prior_document_bytes,
    load_calendar_prior_artifact,
    write_calendar_prior_document,
)
from trading_storage import write_shard_payload


class CalendarPriorArtifactTests(unittest.TestCase):
    def document(self):
        fields = ("dayOfWeek", "hourOfDay")
        table = np.full((7 * 24, ACTION_COUNT), 1 / ACTION_COUNT)
        return build_calendar_prior_document(
            artifact_id="fixture",
            table=table,
            fields=fields,
            estimator={"spec": "hour-by-weekday"},
            oracle_target={"temperature": 0.01},
            training_corpus={"rowsAfterPurge": 100},
            selection_evidence={"calibrationKl": 1.0},
        )

    def test_round_trip_and_calendar_lookup(self) -> None:
        document = self.document()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            write_calendar_prior_document(document, path)
            first = path.read_bytes()
            write_calendar_prior_document(document, path)
            self.assertEqual(first, path.read_bytes())
            artifact = load_calendar_prior_artifact(path)
        calendar = np.zeros((2, len(FEATURE_INDEX)), dtype=np.int16)
        calendar[0, FEATURE_INDEX["dayOfWeek"]] = 1
        calendar[0, FEATURE_INDEX["hourOfDay"]] = 2
        calendar[1, FEATURE_INDEX["dayOfWeek"]] = 4
        calendar[1, FEATURE_INDEX["hourOfDay"]] = 19
        probabilities = artifact.probabilities(calendar)
        self.assertEqual(probabilities.shape, (2, ACTION_COUNT))
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            1,
            atol=2e-6,
        )

    def test_checksum_tampering_is_rejected(self) -> None:
        document = self.document()
        broken = copy.deepcopy(document)
        broken["table"]["values"][0][0] += 0.01
        with self.assertRaisesRegex(ValueError, "probabilities|checksum"):
            artifact_from_document(broken)

    def test_loads_content_addressed_immutable_reference(self) -> None:
        document = self.document()
        with tempfile.TemporaryDirectory() as directory:
            reference = write_shard_payload(
                Path(directory) / "immutable",
                "models/calendar",
                "fixture/model",
                calendar_prior_document_bytes(document),
                sequence={"start": 0, "step": 1, "count": 1, "unit": "index"},
                layout={
                    "encoding": "canonical-json-utf8-v1",
                    "schema": "utc-calendar-table-float32-raw01-policy-v1",
                },
            )
            artifact = load_calendar_prior_artifact(reference)
        self.assertEqual(artifact.fields, ("dayOfWeek", "hourOfDay"))
        self.assertEqual(
            artifact.probabilities_by_cell.shape,
            (7 * 24, ACTION_COUNT),
        )

    def test_checked_in_plan_is_train_only_and_low_dimensional(self) -> None:
        plan_file = (
            Path(__file__).resolve().parent
            / "training-plans/oracle-calendar-prior-hour-weekday-v1.json"
        )
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        validate_artifact_plan(plan)
        self.assertEqual(plan["testPolicy"], "sealed-never-load")
        self.assertEqual(plan["calendarEstimator"]["spec"], "hour-by-weekday")
        self.assertEqual(
            plan["selectionEvidence"]["effectiveCells"],
            168,
        )


if __name__ == "__main__":
    unittest.main()
