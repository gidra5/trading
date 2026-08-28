from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import global_feature_training_dataset as module


class IndexedGlobalFeatureDatasetTest(unittest.TestCase):
    def test_gathers_columns_rows_nonzero_and_spread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            matrix = np.arange(20, dtype=np.float32).reshape(4, 5)
            matrix.tofile(root / "matrix.f32")
            np.asarray([1000, 2000, 3000], dtype="<f8").tofile(dataset / "origins.f64")
            np.asarray([0.0, 0.1, -0.2], dtype="<f4").tofile(dataset / "targets.f32")
            np.asarray([2, 1, 3], dtype="<u4").tofile(dataset / "source-rows.u32")
            np.asarray([0, 1, 2], dtype="u1").tofile(dataset / "splits.u8")
            np.asarray([0, 1, 1], dtype="u1").tofile(dataset / "nonzero.u8")
            np.asarray([1.1, 1.2, 1.3], dtype="<f4").tofile(dataset / "spread.f32")
            manifest = {
                "rows": 3,
                "featureStorage": {
                    "matrix": "matrix.f32", "matrixDtype": "<f4", "matrixShape": [4, 5],
                    "selectedColumnIndices": [4, 1],
                },
                "files": {
                    "origins": {"file": "origins.f64"}, "targets": {"file": "targets.f32"},
                    "sourceRows": {"file": "source-rows.u32"}, "splits": {"file": "splits.u8"},
                    "nonzero": {"file": "nonzero.u8"}, "spread": {"file": "spread.f32"},
                },
            }
            (dataset / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(module, "ROOT", root):
                values = module.IndexedGlobalFeatureDataset(dataset, selection="nonzero")
                self.assertEqual(len(values), 2)
                batch = values.read([1, 0])
                values.close()
            np.testing.assert_array_equal(batch["features"][:, :2], [[19, 16], [9, 6]])
            np.testing.assert_allclose(batch["features"][:, 2], [1.3, 1.2])
            np.testing.assert_allclose(batch["targets"], [-0.2, 0.1])


if __name__ == "__main__":
    unittest.main()
