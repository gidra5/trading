import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from hawkes_forecast_comparison import bootstrap_draws, compare_predictive_losses, origin_blocks, paired_metrics


class PairedForecastTests(unittest.TestCase):
    def test_loss_intervals_match_explicit_unequal_block_resampling_and_reject_misalignment(self):
        seconds = np.array([0, 1, 2, 3, 10, 11, 12])
        reference = np.ones(len(seconds))
        difference = np.array([1., 2., 3., 4., 8., 9., 10.])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / 'reference', root / 'candidate']
            for directory, losses in zip(directories, (reference, reference + difference)):
                directory.mkdir()
                np.savez(directory / 'test_predictive_loss.npz', seconds=seconds, marginal_nll=losses)
                (directory / 'test_one_step_metrics.json').write_text(json.dumps({
                    'predictive_marginal_nll': float(losses.mean()), 'poisson_mean_log_score': 99.}))
            result = compare_predictive_losses(directories, ['A', 'B'], root / 'comparison',
                                              replicates=100, block_seconds=3)[0]
            explicit = bootstrap_draws(seconds, replicates=100, stride=1, block_seconds=3)
            estimates = [difference[draw].mean() for draw in explicit]
            np.testing.assert_allclose(result['paired_block_interval95'], np.quantile(estimates, [.025, .975]))
            self.assertEqual(result['marginal_nll_delta'], difference.mean())
            # A stored filtering loss is sufficient for a marginal-NLL comparison;
            # a reranking cache need not include predictive means or their score.
            (directories[1] / 'test_one_step_metrics.json').write_text(json.dumps({
                'predictive_marginal_nll': float((reference + difference).mean())}))
            cached_result = compare_predictive_losses(directories, ['A', 'B'], root / 'cached',
                                                     replicates=100, block_seconds=3)[0]
            self.assertEqual(cached_result, result)
            np.savez(directories[1] / 'test_predictive_loss.npz', seconds=seconds + 1,
                     marginal_nll=reference + difference)
            with self.assertRaisesRegex(ValueError, 'different scored times'):
                compare_predictive_losses(directories, ['A', 'B'], root / 'bad')
            np.savez(directories[1] / 'test_predictive_loss.npz', seconds=seconds,
                     marginal_nll=np.r_[np.nan, (reference + difference)[1:]])
            with self.assertRaisesRegex(ValueError, 'no survivor conditioning'):
                compare_predictive_losses(directories, ['A', 'B'], root / 'bad')

    def test_blocks_cover_each_origin_once_and_stop_at_gaps(self):
        origins = np.array([0, 900, 1800, 9000, 9900])
        blocks = origin_blocks(origins, block_seconds=1800)
        np.testing.assert_array_equal(np.concatenate(blocks), np.arange(5))
        self.assertTrue(all(np.all(np.diff(origins[block]) == 900) for block in blocks))
        self.assertTrue(all(len(block) <= 2 for block in blocks))

    def test_identical_forecasts_have_zero_paired_differences(self):
        actual = np.array([[1., 2.], [4., 6.], [2., 1.], [10., 4.]])
        prediction = actual + .5
        draws = bootstrap_draws(np.arange(4) * 900, replicates=30, block_seconds=1800)
        result = paired_metrics(actual, prediction, prediction, draws)
        self.assertEqual(result['mae_delta_interval95'], [0., 0.])
        self.assertEqual(result['r2_delta_interval95'], [0., 0.])

    def test_failed_forecast_never_conditions_on_survivors(self):
        actual = np.array([[1., 2.], [4., 6.], [2., 1.]])
        candidate = actual.copy()
        candidate[1] = np.nan
        result = paired_metrics(actual, actual, candidate, (np.arange(3),))
        self.assertFalse(result['comparison_available'])
        self.assertIsNone(result['mae_delta'])


if __name__ == '__main__':
    unittest.main()
