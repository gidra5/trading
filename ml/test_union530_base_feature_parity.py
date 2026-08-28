from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from differentiable_union530_features import (  # noqa: E402
    production59_features,
    reconstruct_global470,
)
from train_feature_augmented_next_return import FeatureMatrixDataset  # noqa: E402
from union530_base_dataset import Union530BaseHistoryDataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/training/datasets/union530-base-history-30d-v1"
GLOBAL = ROOT / "data/training/datasets/global-btc-production-471-next-1s-30d-v1"
PRODUCTION = (
    ROOT
    / "data/training/datasets/next-return-production-basis-history120-full-30d-v1"
)
WORKING = ROOT / "data/runtime-cache/global-btc-expanded-working-set-v4"


def production_parity(dataset: Union530BaseHistoryDataset) -> dict[str, object]:
    split = dataset.split("train", 100)
    rebuilt = production59_features(
        dataset.production_base(torch.device("cpu"), torch.float64)
    ).detach().float().numpy()[dataset.second_indices(split.physical_rows)]
    retained = FeatureMatrixDataset(
        GLOBAL,
        union_history_root=PRODUCTION,
        feature_history_seconds=1,
    )
    expected = np.asarray(
        retained.splits["train"].features[:100], dtype=np.float32
    )[:, :59]
    error = np.abs(expected - rebuilt)
    return {
        "rows": int(expected.shape[0]),
        "maximumAbsoluteError": float(error.max()),
        "meanAbsoluteError": float(error.mean()),
        "worstFeature": dataset.feature_ids[int(error.max(axis=0).argmax())],
        "finiteMismatch": int(np.count_nonzero(
            np.isfinite(expected) != np.isfinite(rebuilt)
        )),
    }


def global_parity(dataset: Union530BaseHistoryDataset) -> dict[str, object]:
    rebuilt = reconstruct_global470(
        dataset.global_base(torch.device("cpu"), torch.float64),
        dataset.global_specs,
    ).detach().float().numpy()
    global_manifest = json.loads((GLOBAL / "dataset.json").read_text(encoding="utf-8"))
    working_manifest = json.loads((WORKING / "manifest.json").read_text(encoding="utf-8"))
    matrix = np.memmap(
        WORKING / working_manifest["file"],
        dtype="<f4",
        mode="r",
        shape=(int(working_manifest["rows"]), int(working_manifest["columns"])),
    )
    columns = np.asarray(
        global_manifest["featureStorage"]["selectedColumnIndices"], dtype=np.int64
    )
    used = np.unique(np.asarray(dataset.minute_source_rows, dtype=np.int64))
    expected = np.asarray(matrix[used[:, None], columns[None, :]], dtype=np.float32)
    actual = rebuilt[used]
    finite_expected = np.isfinite(expected)
    finite_actual = np.isfinite(actual)
    both = finite_expected & finite_actual
    error = np.abs(expected - actual)
    providers = np.asarray([spec["provider"] for spec in dataset.global_specs[:-1]])
    families: dict[str, object] = {}
    for provider in sorted(set(providers)):
        selected = providers == provider
        valid = both[:, selected]
        difference = error[:, selected]
        families[provider] = {
            "maximumAbsoluteError": float(difference[valid].max()),
            "meanAbsoluteError": float(difference[valid].mean()),
            "finiteMismatch": int(np.count_nonzero(
                finite_expected[:, selected] != finite_actual[:, selected]
            )),
        }
    comparable_error = np.where(both, error, np.nan)
    worst_by_feature = np.nanmax(comparable_error, axis=0)
    worst = np.argsort(np.nan_to_num(worst_by_feature, nan=-1))[-20:][::-1]
    return {
        "usedMinuteRows": int(used.size),
        "finiteMismatch": int(np.count_nonzero(finite_expected != finite_actual)),
        "families": families,
        "worstFeatures": [
            {
                "id": dataset.global_specs[int(index)]["id"],
                "maximumAbsoluteError": float(worst_by_feature[index]),
            }
            for index in worst
        ],
    }


def main() -> None:
    dataset = Union530BaseHistoryDataset(BASE)
    print(json.dumps({
        "production59": production_parity(dataset),
        "global470": global_parity(dataset),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
