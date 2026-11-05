import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from horizon_price_forecast import HorizonPriceForecast, fit_horizon_basis
from warm_target_horizon_forecast import CONTRACT, WarmTargetHorizonForecast, load_warm_forecast
from dataclasses import asdict


class WarmHorizonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(13)
        basis = fit_horizon_basis(np.r_[np.zeros(100), np.linspace(-.002, .003, 600)], 32)
        self.head = HorizonPriceForecast(torch.zeros(3), torch.ones(3), basis, (8, 8)).eval()
        # Make the synthetic parent nonconstant, as a trained checkpoint would be.
        with torch.no_grad():
            self.head.output.weight.normal_(0, .1)
        self.model = WarmTargetHorizonForecast(self.head)
        self.x = torch.randn(12, 3)

    def test_initial_full_law_is_identical_at_h1(self):
        old = self.head(self.x)
        new = self.model(self.x, torch.ones(12, dtype=torch.long))
        self.assertTrue(torch.equal(old, new))
        y = torch.linspace(-.001, .001, 12).double(); y[0] = 0
        self.assertTrue(torch.equal(self.head.log_prob_from_logits(old.double(), y),
            self.model.log_prob_from_logits(new, y, torch.ones(12))))
        self.assertTrue(torch.equal(self.head.expectation_from_logits(old),
            self.model.expectation_from_logits(new, torch.ones(12))))

    def test_horizon_weights_train_without_changing_h1(self):
        self.head.requires_grad_(False)
        before = self.model(self.x, 1).detach().clone()
        optimizer = torch.optim.Adam([self.model.horizon_weight], lr=.01)
        h = torch.full((12,), 7200)
        for _ in range(5):
            optimizer.zero_grad()
            loss = -self.model.log_prob_from_logits(self.model(self.x, h), torch.full((12,), .01), h).mean()
            loss.backward(); optimizer.step()
        self.assertGreater(float(self.model.horizon_weight.detach().abs().max()), 0)
        self.assertTrue(torch.equal(before, self.model(self.x, 1)))
        self.assertFalse(torch.equal(self.model(self.x, 1), self.model(self.x, 7200)))

    def test_checkpoint_restores_trained_horizon_weight(self):
        with torch.no_grad():
            self.model.horizon_weight.normal_()
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'warm.pt'
            torch.save({'contract': CONTRACT, 'model': self.model.state_dict(), 'featureMean': self.head.feature_mean,
                'featureStd': self.head.feature_std, 'basis': self.head.basis, 'widths': self.head.widths,
                'horizonRange': asdict(self.model.horizon_range)}, file)
            loaded, _ = load_warm_forecast(file, torch.device('cpu'))
            self.assertTrue(torch.equal(self.model(self.x, 7200), loaded(self.x, 7200)))


if __name__ == '__main__':
    unittest.main()
