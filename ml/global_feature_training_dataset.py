from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from global_feature_registry import ROOT


class IndexedGlobalFeatureDataset:
    """Batch-oriented reader for compact indexed global-feature datasets."""

    def __init__(self, directory: Path, *, selection: str = "all") -> None:
        self.directory = directory.resolve()
        self.manifest = json.loads((self.directory / "dataset.json").read_text(encoding="utf-8"))
        if selection not in {"all", "nonzero"}:
            raise ValueError("selection must be 'all' or 'nonzero'")
        self.selection = selection
        self.total_rows = int(self.manifest["rows"])
        files = self.manifest["files"]
        self.origins = np.memmap(
            self.directory / files["origins"]["file"], dtype="<f8", mode="r", shape=(self.total_rows,)
        )
        self.targets = np.memmap(
            self.directory / files["targets"]["file"], dtype="<f4", mode="r", shape=(self.total_rows,)
        )
        self.source_rows = np.memmap(
            self.directory / files["sourceRows"]["file"], dtype="<u4", mode="r", shape=(self.total_rows,)
        )
        self.splits = np.memmap(
            self.directory / files["splits"]["file"], dtype="u1", mode="r", shape=(self.total_rows,)
        )
        self.nonzero = np.memmap(
            self.directory / files["nonzero"]["file"], dtype="u1", mode="r", shape=(self.total_rows,)
        )
        self.rows = (
            np.arange(self.total_rows, dtype=np.int64)
            if selection == "all" else np.flatnonzero(self.nonzero)
        )
        storage = self.manifest["featureStorage"]
        matrix_rows, matrix_columns = map(int, storage["matrixShape"])
        self.matrix = np.memmap(
            ROOT / storage["matrix"], dtype=storage["matrixDtype"], mode="r",
            shape=(matrix_rows, matrix_columns),
        )
        self.columns = np.asarray(storage["selectedColumnIndices"], dtype=np.int64)
        spread = files.get("spread")
        self.spread = None if spread is None else np.memmap(
            self.directory / spread["file"], dtype="<f4", mode="r", shape=(self.total_rows,)
        )

    def __len__(self) -> int:
        return int(self.rows.size)

    def close(self) -> None:
        for value in (
            self.origins, self.targets, self.source_rows, self.splits,
            self.nonzero, self.matrix, self.spread,
        ):
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self) -> "IndexedGlobalFeatureDataset":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read(self, indices: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        logical = np.asarray(indices, dtype=np.int64)
        physical = self.rows[logical]
        source = np.asarray(self.source_rows[physical], dtype=np.int64)
        features = np.asarray(self.matrix[source[:, None], self.columns[None, :]], dtype=np.float32)
        if self.spread is not None:
            features = np.column_stack((features, np.asarray(self.spread[physical], dtype=np.float32)))
        return {
            "features": features,
            "targets": np.asarray(self.targets[physical], dtype=np.float32),
            "origins": np.asarray(self.origins[physical], dtype=np.float64),
            "splits": np.asarray(self.splits[physical], dtype=np.uint8),
        }
