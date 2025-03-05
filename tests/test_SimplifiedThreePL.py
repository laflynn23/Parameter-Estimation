import unittest
import numpy as np
from SimplifiedThreePL import SimplifiedThreePL
from Experiment import Experiment
from SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        self.experiment = Experiment()
        for _ in range(5):
            sdt = SignalDetection(25, 25, 25, 25)
            self.experiment.add_condition(sdt)
        self.model = SimplifiedThreePL(self.experiment)

    def test_summary(self):
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 500)
        self.assertEqual(summary["n_correct"], 250)
        self.assertEqual(summary["n_incorrect"], 250)
        self.assertEqual(summary["n_conditions"], 5)

    def test_invalid_initialization(self):
        exp_invalid = Experiment()
        for _ in range(4):
            sdt = SignalDetection(25, 25, 25, 25)
            exp_invalid.add_condition(sdt)
        with self.assertRaises(ValueError):
            _ = SimplifiedThreePL(exp_invalid)

    def test_predict_range(self):
        params = (1.0, 0.0) 
        predictions = self.model.predict(params)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_predict_effects(self):
        params_low = (1.0, -2.0) 
        params_high = (1.0, 2.0) 
        pred_low = self.model.predict(params_low)
        pred_high = self.model.predict(params_high)
        self.assertTrue(np.all(pred_high >= pred_low))
        
        params = (1.0, 0.0)
        predictions = self.model.predict(params)
        self.assertLess(predictions[0], predictions[-1])
        
    def test_get_parameters_before_fit(self):
        with self.assertRaises(RuntimeError):
            _ = self.model.get_discrimination()
        with self.assertRaises(RuntimeError):
            _ = self.model.get_base_rate()

    def test_fit_improves_likelihood(self):
        init_params = [1.0, 0.0]
        nll_initial = self.model.negative_log_likelihood(init_params)
        self.model.fit()
        fitted_params = [self.model.get_discrimination(),
                         np.log(self.model.get_base_rate() / (1 - self.model.get_base_rate()))]
        nll_fitted = self.model.negative_log_likelihood(fitted_params)
        self.assertLess(nll_fitted, nll_initial)

    def test_integration(self):
        experiment = Experiment()
        experiment.add_condition(SignalDetection(28, 22, 23, 27))
        experiment.add_condition(SignalDetection(30, 20, 20, 30))
        experiment.add_condition(SignalDetection(38, 12, 13, 37))
        experiment.add_condition(SignalDetection(45, 5, 5, 45))
        experiment.add_condition(SignalDetection(48, 2, 3, 47))
        
        model = SimplifiedThreePL(experiment)
        model.fit()
        predictions = model.predict((model.get_discrimination(), model._logit_base_rate))
        observed = [55/100, 60/100, 75/100, 90/100, 95/100]
        for pred, obs in zip(predictions, observed):
            self.assertAlmostEqual(pred, obs, delta=0.1)

    def test_stability_of_fit(self):
        self.model.fit()
        a1 = self.model.get_discrimination()
        c1 = self.model.get_base_rate()
        self.model.fit()
        a2 = self.model.get_discrimination()
        c2 = self.model.get_base_rate()
        self.assertAlmostEqual(a1, a2, places=3)
        self.assertAlmostEqual(c1, c2, places=3)

if __name__ == '__main__':
    unittest.main()