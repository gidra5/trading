from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_joint_price_oracle import atomic_json, replace_file_with_retry


class TrainJointPriceOracleAtomicJsonTest(unittest.TestCase):
    def test_atomic_json_retries_transient_permission_error(self) -> None:
        real_replace = os.replace
        replace_calls = 0

        def replace_after_transient_lock(source: Path, destination: Path) -> None:
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 1:
                raise PermissionError("locked")
            real_replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "status.json"
            with patch(
                "train_joint_price_oracle.os.replace",
                side_effect=replace_after_transient_lock,
            ), patch("train_joint_price_oracle.time.sleep") as sleep:
                atomic_json({"stage": "running"}, destination)

            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")),
                {"stage": "running"},
            )
            self.assertEqual(replace_calls, 2)
            sleep.assert_called_once_with(0.01)

    def test_persistent_permission_error_is_raised_after_bound(self) -> None:
        with patch(
            "train_joint_price_oracle.os.replace",
            side_effect=PermissionError("still locked"),
        ) as replace, patch("train_joint_price_oracle.time.sleep") as sleep:
            with self.assertRaisesRegex(PermissionError, "still locked"):
                replace_file_with_retry(
                    Path("status.json.tmp"),
                    Path("status.json"),
                    attempts=3,
                )

        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_non_permission_error_is_not_retried(self) -> None:
        with patch(
            "train_joint_price_oracle.os.replace",
            side_effect=OSError("disk failure"),
        ) as replace, patch("train_joint_price_oracle.time.sleep") as sleep:
            with self.assertRaisesRegex(OSError, "disk failure"):
                replace_file_with_retry(
                    Path("status.json.tmp"),
                    Path("status.json"),
                )

        replace.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
