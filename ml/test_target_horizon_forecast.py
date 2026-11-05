import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from horizon_price_forecast import fit_horizon_basis
from target_horizon_forecast import CONTRACT, HorizonRange, TargetHorizonForecast, complete_origins, load_target_horizon_forecast
from train_target_horizon_forecast import targets


class TargetHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.basis = fit_horizon_basis(np.r_[np.zeros(100), np.linspace(-.001, .001, 600)], 32)

    def model(self):
        torch.manual_seed(7)
        return TargetHorizonForecast(torch.zeros(2), torch.ones(2), self.basis, widths=(8,)).eval()

    def test_sampling_matches_integrated_tanh_derivative(self):
        law = HorizonRange()
        p = law.probabilities()
        self.assertAlmostEqual(p.sum(), 1., places=12)
        self.assertTrue(np.all(p > 0))
        h = law.sample(np.random.default_rng(10), 400000)
        for cutoff in [60, 7200, 21600, 43200]:
            self.assertLess(abs(np.mean(h <= cutoff)-p[:cutoff].sum()), .003)
        self.assertLess(abs(np.tanh(7200/21600)/(7200/21600)-1), .036)

    def test_purge_keeps_common_complete_origins(self):
        actual = complete_origins(np.array([1, 4, 8, 13]), 14, HorizonRange(5, 6))
        np.testing.assert_array_equal(actual, [1, 4, 8])

    def test_targets_use_exact_endpoints_and_zero_atom(self):
        data = {'indices': np.array([0, 1]), 'prices': np.array([100., 110., 100., 121.])}
        np.testing.assert_allclose(targets(data, np.array([2, 2])), [0., np.log(1.1)])
        np.testing.assert_allclose(targets(data, np.array([1, 1])), [np.log(1.1), np.log(100/110)])

    def test_reject_invalid_horizons(self):
        model = self.model()
        for h in [0, -1, 86401, 1.2, float('nan'), True]:
            with self.assertRaises(ValueError):
                model(torch.zeros(2, 2), torch.tensor(h))

    def test_mixed_law_jacobian_and_atom(self):
        model = self.model()
        logits = model(torch.zeros(2, 2), torch.tensor([1, 4]))
        y = torch.tensor([0., .0002], dtype=torch.float64)
        h = torch.tensor([1, 4])
        ll = model.log_prob_from_logits(logits, y, h)
        reference = model.head.log_prob_from_logits(logits.double(), y/torch.sqrt(h.double()))
        torch.testing.assert_close(ll, reference-torch.tensor([0., np.log(2.)]))
        torch.testing.assert_close(ll[0], torch.nn.functional.logsigmoid(logits[0, 0].double()))

    def test_full_law_normalization_and_causal_interface(self):
        model = self.model()
        x, h = torch.randn(3, 2), torch.tensor([1, 7200, 86400])
        a = model(x, h)
        z, masses = model.probabilities(x, h)
        torch.testing.assert_close(z+(1-z)*masses.sum(dim=1), torch.ones(3, dtype=torch.float64))
        # Scoring different futures never changes the forecast.
        model.log_prob_from_logits(a, torch.zeros(3), h)
        model.log_prob_from_logits(a, torch.ones(3)*.01, h)
        torch.testing.assert_close(a, model(x, h))
        self.assertTrue(np.isfinite(model.quantile_returns(masses.numpy(), h.numpy(), 9)).all())

    def test_gradient_and_checkpoint_roundtrip(self):
        model = self.model()
        x, h = torch.randn(2, 2), torch.tensor([600, 7200])
        logits = model(x, h)
        loss = -model.log_prob_from_logits(logits, torch.tensor([.001, -.01]), h).mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'test.pt'
            torch.save({'contract': CONTRACT, 'model': model.state_dict(), 'basis': self.basis,
                'featureMean': torch.zeros(2), 'featureStd': torch.ones(2), 'widths': [8],
                'horizonRange': {'scale_seconds': 21600., 'maximum_seconds': 86400}}, file)
            restored, _ = load_target_horizon_forecast(file, torch.device('cpu'))
            torch.testing.assert_close(logits, restored(x, h))


if __name__ == '__main__':
    unittest.main()
