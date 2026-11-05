import unittest
import numpy as np
import torch
from horizon_price_forecast import HorizonPriceForecast, fit_horizon_basis
from refit_ensemble_distribution_heads import forecast_law
from evaluate_refitted_ensemble import regime_weights, regime_predictions


class RefitEnsembleTests(unittest.TestCase):
    def test_head_predictions_do_not_depend_on_scored_future(self):
        torch.set_num_threads(2)
        torch.manual_seed(10)
        y = np.r_[np.zeros(100), np.linspace(-.001, .001, 600)]
        model = HorizonPriceForecast(torch.zeros(3), torch.ones(3), fit_horizon_basis(y), (8,)).eval()
        x = np.array([[.1, -.2, .3], [-.4, .5, -.6]], dtype=np.float32)
        a = forecast_law(model, x, np.array([0., .0001]), torch.device('cpu'))
        b = forecast_law(model, x, np.array([-.0003, 0.]), torch.device('cpu'))
        for key in ['mean', 'zeroProbability', 'componentMasses']:
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(a['componentMasses'].sum(axis=1), 1, atol=1e-12)
        gross = model.active_gross_mean(a['componentMasses'])
        self.assertTrue(np.isfinite(gross).all())
        np.testing.assert_allclose(a['logLikelihood'][0], np.log(a['zeroProbability'][0]))

    def test_regime_prediction_uses_forecasts_only(self):
        x = np.array([[1., 2.], [-1., 2.], [-2., -3.], [3., -1.]])*1e-4
        y = np.array([1., 2., -2., -1.])*1e-4
        weights = regime_weights(x, y)
        mean, ll = regime_predictions(x, np.log(np.full_like(x, .5)), weights)
        np.testing.assert_allclose(mean, y, atol=1e-12)
        np.testing.assert_allclose(ll, np.log(.5))


if __name__ == '__main__':
    unittest.main()
