from __future__ import annotations

import unittest

import numpy as np

from global_feature_candidate_axis import (
    BaseRecentBatchProvider,
    DenseMinuteBatchProvider,
    FundingGridBatchProvider,
    LongUniqueMinuteBatchProvider,
    OneSecondTechnicalBatchProvider,
    PublicExternalBatchProvider,
    RepresentativeCrossAssetBatchProvider,
    SpectralMinuteBatchProvider,
    SpectralSecondBatchProvider,
    dense_grid,
    quantize_batch_fast,
)
from global_feature_registry import coordinate_id


class GlobalFeatureCandidateAxisTest(unittest.TestCase):
    def test_vectorized_quantizer_is_equal_mass_and_reserves_missing_state(self) -> None:
        values = np.asarray([
            [0.0, 10.0], [1.0, 11.0], [2.0, np.nan], [3.0, 13.0],
            [4.0, 14.0], [5.0, 15.0], [6.0, 16.0], [7.0, 17.0],
        ])
        states, edges, arities = quantize_batch_fast(values, np.ones(8, dtype=bool))
        self.assertEqual(arities, [4, 5])
        self.assertEqual(states[2, 1], 4)
        self.assertEqual(len(edges), 2)
        self.assertEqual(set(states[:, 0]), {0, 1, 2, 3})

    def test_dense_grid_count(self) -> None:
        grid = dense_grid()
        base = len(grid["rsi"]) + len(grid["ema"]) * (1 + 2 * len(grid["horizons"]))
        self.assertEqual(base, 267)
        self.assertEqual(base * len(grid["lags"]), 3_471)

    def test_one_asset_stream_has_expected_unique_coordinates(self) -> None:
        provider = DenseMinuteBatchProvider(batch_size=512, limit_assets=1)
        ids: list[str] = []
        rows = None
        for batch in provider.quantized_batches():
            ids.extend(group.id for group in batch.groups)
            rows = batch.states.shape[0]
            self.assertEqual(batch.states.dtype, np.uint8)
            self.assertEqual(batch.states.shape[1], len(batch.groups))
        self.assertEqual(rows, 42_901)
        self.assertEqual(len(ids), 3_471)
        self.assertEqual(len(set(ids)), len(ids))

    def test_asset_binding_is_part_of_coordinate_identity(self) -> None:
        provider = DenseMinuteBatchProvider(batch_size=1_024, limit_assets=2)
        ids = [group.id for batch in provider.quantized_batches() for group in batch.groups]
        self.assertEqual(len(ids), 2 * 3_471)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertNotEqual(ids[0].split("/")[1], ids[3_471].split("/")[1])

    def test_selected_stream_reconstructs_only_requested_coordinate(self) -> None:
        probe = DenseMinuteBatchProvider(limit_assets=1)
        feature_id = coordinate_id(
            probe.assets[0]["asset"],
            "asset",
            "binance-preferred",
            "1m",
            f"rsi-{probe.grid['rsi'][0]}m",
        )
        provider = DenseMinuteBatchProvider(limit_assets=1, selected_ids={feature_id})
        batches = list(provider.quantized_batches())
        self.assertEqual([group.id for batch in batches for group in batch.groups], [feature_id])
        self.assertEqual(batches[0].states.shape, (42_901, 1))

    def test_existing_recent_inventory_has_147_unique_canonical_coordinates(self) -> None:
        provider = BaseRecentBatchProvider()
        batches = list(provider.quantized_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), 147)
        self.assertEqual(len(set(ids)), 147)
        self.assertEqual(batches[0].states.shape, (42_901, 147))

        selected = BaseRecentBatchProvider(selected_ids={ids[0]})
        selected_ids = [group.id for batch in selected.quantized_batches() for group in batch.groups]
        self.assertEqual(selected_ids, [ids[0]])

    def test_representative_cross_asset_stream_matches_its_catalog(self) -> None:
        provider = RepresentativeCrossAssetBatchProvider(batch_size=64, limit_assets=1)
        batches = list(provider.raw_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), provider.coordinate_count)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(ids)
        self.assertTrue(all(batch.values.shape[0] == 42_901 for batch in batches))

    def test_long_unique_provider_emits_asset_and_global_coordinates(self) -> None:
        provider = LongUniqueMinuteBatchProvider(batch_size=4, limit_assets=1)
        batches = list(provider.raw_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), 9)
        self.assertEqual(len(set(ids)), 9)
        self.assertEqual(provider.coordinate_count, 9)
        self.assertIn("general/global/calendar/known/utc-hour-sin", ids)

    def test_spectral_minute_provider_emits_all_102_unique_definitions(self) -> None:
        provider = SpectralMinuteBatchProvider(batch_size=128, limit_assets=1)
        batches = list(provider.raw_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), 102)
        self.assertEqual(len(set(ids)), 102)
        self.assertEqual(provider.coordinate_count, 102)
        self.assertEqual(batches[0].values.shape, (42_901, 102))
        selected_id = coordinate_id(
            provider.assets[0]["asset"], "asset", "binance-preferred", "1m",
            provider.definitions[0]["id"],
        )
        selected = SpectralMinuteBatchProvider(limit_assets=1, selected_ids={selected_id})
        selected_batch = list(selected.raw_batches())[0]
        self.assertEqual([group.id for group in selected_batch.groups], [selected_id])
        np.testing.assert_allclose(
            selected_batch.values[:, 0], batches[0].values[:, 0], equal_nan=True
        )

    def test_public_external_export_has_675_unique_canonical_coordinates(self) -> None:
        provider = PublicExternalBatchProvider(batch_size=200)
        batches = list(provider.raw_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), 675)
        self.assertEqual(len(set(ids)), 675)
        self.assertEqual(provider.coordinate_count, 675)
        self.assertEqual(sum(batch.values.shape[1] for batch in batches), 675)

    def test_full_funding_grid_matches_registry_count_and_can_select_one(self) -> None:
        provider = FundingGridBatchProvider()
        self.assertEqual(provider.coordinate_count, 4_242)
        selected_id = coordinate_id(
            provider.assets[0]["asset"], "asset", "binance-usdm", "funding-event", "funding-level"
        )
        selected = FundingGridBatchProvider(selected_ids={selected_id})
        batches = list(selected.raw_batches())
        self.assertEqual([group.id for batch in batches for group in batch.groups], [selected_id])
        self.assertEqual(batches[0].values.shape, (42_901, 1))

    def test_one_second_technical_provider_emits_all_166_definitions(self) -> None:
        provider = OneSecondTechnicalBatchProvider(limit_assets=1)
        batches = list(provider.raw_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), 166)
        self.assertEqual(len(set(ids)), 166)
        self.assertEqual(batches[0].values.shape, (42_901, 166))

    def test_one_second_spectral_provider_emits_all_102_definitions(self) -> None:
        provider = SpectralSecondBatchProvider(limit_assets=1)
        batches = list(provider.raw_batches())
        ids = [group.id for batch in batches for group in batch.groups]
        self.assertEqual(len(ids), 102)
        self.assertEqual(len(set(ids)), 102)
        self.assertEqual(batches[0].values.shape, (42_901, 102))
        selected_id = provider.feature_id(
            str(provider.assets[0]["asset"]), provider.definitions[0]
        )
        selected = SpectralSecondBatchProvider(limit_assets=1, selected_ids={selected_id})
        selected_batch = list(selected.raw_batches())[0]
        self.assertEqual([group.id for group in selected_batch.groups], [selected_id])
        np.testing.assert_allclose(
            selected_batch.values[:, 0], batches[0].values[:, 0], equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
