import itertools
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from hawkes_final_test import table
from hawkes_report import LABELS, report
from hawkes_research import HORIZONS, save_json


class ResearchReportTests(unittest.TestCase):
    def fixture(self, root):
        save_json(root / 'evaluation-complete.json', {
            'all_five_models_evaluated': True, 'test_evaluated': True,
            'parameters_changed_after_freeze': False, 'forecast_origins_identical': True,
            'protocol': {'historical_test_inspection': 'Earlier exploratory runs inspected the test period.',
                         'filter_particles': 2048, 'future_paths': 2048, 'stride_seconds': 900}})
        decision = root / 'decision.json'
        save_json(decision, {'keep_models': ['M0'], 'rationale': 'Fixture decision only.',
                            'evidence_review': 'Synthetic control fails recovery; no physical latent identity.'})
        table(root / 'final_test_metrics.csv', [
            dict(model=label, split='test', predictive_marginal_nll=5., total_mae=4., total_r2=.1,
                 buy_mae=2., sell_mae=3.) for label in LABELS])
        table(root / 'horizon_metrics.csv', [
            dict(model=label, split='test', horizon_seconds=h, origins=5, total_mae=123., total_r2=.2,
                 total_correlation=.3, failed_origin_fraction=.2 if label == 'M2L' and h == 3600 else 0.)
            for label in LABELS for h in HORIZONS])
        table(root / 'stability.csv', [dict(model=label, spectral_radius=.9, global_branching_bound=False,
                                            one_hour_failed_origin_fraction=.2 if label == 'M2L' else 0.) for label in LABELS])
        table(root / 'latent_diagnostics.csv', [
            dict(model=label, filtered_current_latent_intensity_mean=.1, previous_latent_count_posterior_mean=.1,
                 previous_latent_active_probability_mean=.05,
                 fraction_bins_previous_latent_active_probability_gt_0_01=.2,
                 physical_latent_identifiability_established=False) for label in LABELS[2:]])
        parameters = [dict(model=label, kind='exponential', source='S', target='B', integrated_mass=.15,
                           tau_seconds=37.125) for label in LABELS]
        parameters += [dict(model='M2L', kind='modulation', mode_index=0, anchor_index=0, value=.2,
                            coefficient_on_raw_slow=.01234),
                       dict(model='M2LZ', kind='diffusion_loading', diffusion_index=0, target='S',
                            tau_seconds=3599.25, value=-1.23456789)]
        table(root / 'fitted_scales.csv', parameters)
        table(root / 'model_selection.csv', [dict(family=label, operation='fixture') for label in LABELS])
        comparison = root / 'comparison'
        comparison.mkdir()
        save_json(comparison / 'paired_one_step_differences.json', {
            'protocol': {'split': 'test'}, 'comparisons': [
                dict(reference=a, candidate=b, scored_seconds=240, marginal_nll_delta=0.,
                     paired_block_interval95=[-.1, .1]) for a, b in itertools.combinations(LABELS, 2)]})
        save_json(comparison / 'paired_horizon_differences.json', {
            'protocol': {'split': 'test'}, 'comparisons': [
                dict(reference=a, candidate=b, horizon_seconds=3600, mae_delta=None,
                     mae_delta_interval95=None, r2_delta=None, r2_delta_interval95=None)
                for a, b in itertools.combinations(LABELS, 2)]})
        for label in LABELS:
            (root / label).mkdir()
            save_json(root / label / 'frozen-model.json', {'test_fixture': True})
            save_json(root / label / 'selection-audit.json', {'test_fixture': True})
            (root / label / 'architecture-selection.csv').write_text('family,operation\n' + label + ',fixture\n')
            if label in LABELS[2:]:
                seconds = np.r_[np.arange(120), np.arange(300, 420)]
                np.savez(root / label / 'test_latent_trace.npz', seconds=seconds,
                         filtered_current_latent_intensity=np.full(len(seconds), .1),
                         previous_latent_count_posterior_mean=np.full(len(seconds), .08),
                         previous_latent_active_probability=np.full(len(seconds), .05))
        return decision

    def test_report_retains_failures_parameters_and_scientific_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = self.fixture(root)
            output = report(root, decision, root / 'report.md')
            text = output.read_text(encoding='utf-8')
            self.assertIn('| 3600 | 5 | N/A | N/A | N/A | N/A | 0.2 |', text)
            self.assertIn('coefficient_on_raw_slow', text)
            self.assertIn('-1.2345679', text)
            self.assertIn('Earlier exploratory runs inspected', text)
            self.assertIn('do not prove a global optimum', text)
            self.assertIn('not exact multistep marginal likelihoods', text)
            provenance = json.loads(output.with_suffix('.provenance.json').read_text())
            self.assertFalse(provenance['observations_opened'])
            self.assertTrue(all(len(entry['sha256']) == 64 for entry in provenance['inputs']))

    def test_incomplete_evaluation_and_missing_horizons_cannot_be_reported_as_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = self.fixture(root)
            completion = root / 'evaluation-complete.json'
            value = json.loads(completion.read_text())
            save_json(completion, dict(value, all_five_models_evaluated=False))
            with self.assertRaisesRegex(ValueError, 'completed frozen'):
                report(root, decision, root / 'report.md')
            save_json(completion, value)
            horizon_path = root / 'horizon_metrics.csv'
            lines = horizon_path.read_text().splitlines()
            horizon_path.write_text('\n'.join(lines[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'forecast horizons'):
                report(root, decision, root / 'report.md')
            self.assertFalse((root / 'report.md').exists())


if __name__ == '__main__':
    unittest.main()
