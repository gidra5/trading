from __future__ import annotations

import unittest

from global_feature_registry import (
    build_registry,
    canonical_id,
    coordinate_id,
    feature_level,
    prediction_market_asset,
    source_feature_inventory,
)


class GlobalFeatureRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry, cls.assets = build_registry()
        cls.source_inventory = source_feature_inventory(cls.assets)

    def test_availability_sets_match_download_manifests(self) -> None:
        self.assertEqual(len(self.assets["universe"]), 261)
        self.assertEqual(len(self.assets["preferred1m99"]), 257)
        self.assertEqual(len(self.assets["spot1m99"]), 140)
        self.assertEqual(len(self.assets["usdm1m99"]), 233)
        self.assertEqual(len(self.assets["spot1s95"]), 140)
        self.assertEqual(len(self.assets["usdmMetrics95"]), 196)
        self.assertEqual(len(self.assets["usdmBook95"]), 232)

    def test_dense_grid_includes_all_declared_lags(self) -> None:
        self.assertEqual(self.registry.raw_by_inventory["dense-minute-indicators"], 3_471 * 257)

    def test_general_calendar_is_not_multiplied_by_asset_count(self) -> None:
        hour_sin = canonical_id("general", "calendar", "known", "utc-hour-sin")
        hour_cos = canonical_id("general", "calendar", "known", "utc-hour-cos")
        self.assertEqual(self.registry.templates[hour_sin].subjects, {"GLOBAL"})
        self.assertEqual(self.registry.templates[hour_cos].subjects, {"GLOBAL"})

    def test_known_aliases_deduplicate(self) -> None:
        rsi = canonical_id("asset", "binance-preferred", "1m", "rsi-2m")
        self.assertIn("dense-minute-indicators:rsi-2m", self.registry.templates[rsi].aliases)
        self.assertGreater(len(self.registry.templates[rsi].inventories), 1)
        self.assertEqual(self.registry.templates[rsi].source_policy, "candle-derived")
        fast_rsi = canonical_id("asset", "binance-spot", "1s", "rsi-2s")
        self.assertGreater(len(self.registry.templates[fast_rsi].inventories), 1)
        eth_return = canonical_id("asset", "binance-preferred", "1m", "return-1m")
        self.assertIn("cross-market-public", self.registry.templates[eth_return].inventories)
        self.assertLess(self.registry.unique_coordinates, self.registry.raw_coordinates)

    def test_incumbent_catalog_is_complete(self) -> None:
        self.assertEqual(self.registry.raw_by_inventory["representative-cross-asset"], 31_043)

    def test_summary_reconciles_scope_and_production_eligibility(self) -> None:
        summary = self.registry.summary()
        self.assertEqual(summary["rawCoordinates"], 1_001_102)
        self.assertEqual(summary["uniqueCoordinates"], 991_015)
        self.assertEqual(summary["duplicateLedgerCoordinates"], 10_087)
        self.assertEqual(summary["assetSpecificCoordinates"], 989_853)
        self.assertEqual(summary["generalCoordinates"], 1_162)
        self.assertEqual(summary["basicAssetSpecificCandidateTemplates"], 10)
        self.assertEqual(summary["derivedAssetSpecificCandidateTemplates"], 4_457)
        self.assertEqual(summary["basicGeneralCandidateTemplates"], 66)
        self.assertEqual(summary["derivedGeneralCandidateTemplates"], 1_096)
        self.assertEqual(summary["basicAssetSpecificCandidateCoordinates"], 245)
        self.assertEqual(summary["derivedAssetSpecificCandidateCoordinates"], 989_608)
        self.assertEqual(summary["basicGeneralCandidateCoordinates"], 66)
        self.assertEqual(summary["derivedGeneralCandidateCoordinates"], 1_096)
        self.assertEqual(summary["robust30dCoordinates"], 990_793)
        self.assertEqual(summary["shortWindowCoordinates"], 222)
        self.assertEqual(summary["causalRobust30dCoordinates"], 990_266)
        self.assertEqual(summary["nonPointInTimeRobust30dCoordinates"], 527)

    def test_basic_derived_taxonomy_is_explicit(self) -> None:
        self.assertEqual(feature_level("spot-flow-last-side-1s"), "basic")
        self.assertEqual(feature_level("funding-level"), "basic")
        self.assertEqual(feature_level("macro-cpiaucsl-level"), "basic")
        self.assertEqual(feature_level("funding-absolute-level"), "derived")
        self.assertEqual(feature_level("open-interest-log-level"), "derived")
        self.assertEqual(feature_level("futures-basis-level"), "derived")
        self.assertEqual(feature_level("utc-hour-sin"), "derived")

    def test_source_inventory_keeps_definitions_and_asset_bindings_separate(self) -> None:
        categories = {
            row["id"]: row for row in self.source_inventory["assetSpecific"]["categories"]
        }
        funding = next(
            row for row in categories["futures-stats"]["fieldSets"]
            if row["id"] == "settled-usdm-funding"
        )
        self.assertEqual(funding["fieldDefinitions"], 1)
        self.assertEqual(funding["assetsAvailable"], 236)
        self.assertEqual(funding["expandedAssetFieldBindings"], 236)
        options = categories["options-stats"]["fieldSets"][0]
        self.assertEqual(options["fieldDefinitions"], 20)
        self.assertEqual(options["assetsAvailable"], 1)

    def test_prediction_market_hierarchy_counts_dynamic_contract_fields(self) -> None:
        prediction = self.source_inventory["predictionMarket"]
        self.assertEqual(prediction["identity"]["persistent"], "series")
        self.assertEqual(prediction["fieldSchemas"]["normalizedForCurrentAxis"]["subfeatureDefinitions"], 7)
        asset = prediction["assetSpecific"]
        self.assertEqual(asset["recognizedAssets"], 10)
        self.assertEqual(asset["window"], {
            "persistentSeries": 78, "events": 34_836, "contracts": 95_223,
        })
        self.assertEqual(asset["snapshot"]["openContracts"], 1_008)
        self.assertEqual(asset["snapshot"]["potentialContractMetadataFieldSlots"], 25_200)
        self.assertEqual(asset["snapshot"]["observedContracts1m"], 591)
        self.assertEqual(asset["snapshot"]["observedContracts1s"], 82)
        self.assertEqual(asset["snapshot"]["persistentSeriesFieldChannels"], 546)
        self.assertEqual(asset["snapshot"]["openContractFieldSlots"], 7_056)
        self.assertEqual(asset["snapshot"]["observedContractFieldSlotsUpperBound1m"], 4_137)
        general = prediction["generalEvents"]
        self.assertEqual(general["window"], {
            "persistentSeries": 1_832, "events": 9_031, "contracts": 89_835,
        })
        self.assertEqual(general["snapshot"]["observedContracts1m"], 12)
        self.assertEqual(general["snapshot"]["persistentSeriesFieldChannels"], 10_374)
        self.assertEqual(general["snapshot"]["openContractFieldSlots"], 120_127)
        self.assertEqual(general["snapshot"]["observedContractFieldSlotsUpperBound1m"], 84)
        self.assertEqual(general["snapshot"]["potentialContractMetadataFieldSlots"], 429_025)

    def test_prediction_market_asset_identity_is_shared_with_axis_builder(self) -> None:
        self.assertEqual(prediction_market_asset({
            "series": "KXBTC15M", "seriesTitle": "Bitcoin price up",
        }), "BTC")
        self.assertEqual(prediction_market_asset({
            "series": "KXTOKENLAUNCH", "seriesTitle": "Who launches a token?",
        }), "CRYPTO-OTHER")

    def test_coordinate_identity_contains_concrete_subject(self) -> None:
        eth = coordinate_id("ETH", "asset", "binance-preferred", "1m", "return-1m")
        sol = coordinate_id("SOL", "asset", "binance-preferred", "1m", "return-1m")
        self.assertNotEqual(eth, sol)

    def test_non_ascii_subjects_have_distinct_nonempty_coordinate_ids(self) -> None:
        ids = {
            coordinate_id(subject, "asset", "binance-preferred", "1m", "return-1m")
            for subject in ("龙虾", "币安人生", "我踏马来了")
        }
        self.assertEqual(len(ids), 3)
        self.assertTrue(all("asset//" not in feature_id for feature_id in ids))


if __name__ == "__main__":
    unittest.main()
