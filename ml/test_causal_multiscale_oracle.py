from __future__ import annotations

import unittest

import torch

from return_oracle_decoder_screen import LearnedRadiusShrinkingDecoder
from train_causal_multiscale_oracle import MinuteExample, source_examples


class CausalMultiscaleOracleTest(unittest.TestCase):
    def test_learned_radius_decoder_accepts_rich_feature_width(self) -> None:
        model = LearnedRadiusShrinkingDecoder(
            torch.zeros(771), torch.ones(771), dropout=0, output_count=101
        )
        self.assertEqual(model(torch.zeros(2, 771)).shape, (2, 101))

    def test_source_examples_purges_future_hour_at_run_end(self) -> None:
        manifest = {
            "shards": [{
                "split": "train",
                "date": "2026-01-01",
                "count": 7_200,
                "featureRowOffset": 0,
                "predictionTimeStart": 1_000,
            }]
        }
        values = source_examples(manifest)
        self.assertEqual(len(values), 105)
        self.assertEqual(values[0], MinuteExample(
            timestamp=60_000,
            split="train",
            date="2026-01-01",
            feature_row=59,
            target_row=1,
        ))


if __name__ == "__main__":
    unittest.main()
