import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        """Setup for experiment-based and single-item models."""
        # Experiment-based setup
        self.experiment = Experiment()
        for _ in range(5):
            sdt = SignalDetection(25, 25, 25, 25)
            self.experiment.add_condition(sdt)
        self.model_exp = SimplifiedThreePL(self.experiment)

        # Single-item setup
        self.model_single = SimplifiedThreePL(difficulty=0.5, discrimination=1.2, base_rate=0.2)

    def test_probability_output(self):
        """Ensure probability is between 0 and 1 for single-item model."""
        result = self.model_single.probability(1.0)
        self.assertTrue(0 <= result <= 1)

    def test_invalid_guessing_param(self):
        """Ensure base rate is valid."""
        with self.assertRaises(ValueError):
            SimplifiedThreePL(difficulty=1.0, discrimination=1.0, base_rate=1.5)

    def test_invalid_discrimination_param(self):
        """Ensure discrimination parameter is positive."""
        with self.assertRaises(ValueError):
            SimplifiedThreePL(difficulty=1.0, discrimination=-1.0, base_rate=0.2)

    def test_summary(self):
        summary = self.model_exp.summary()
        self.assertEqual(summary["n_total"], 500)

    def test_predict_range(self):
        """Fit model before testing `predict()` to avoid RuntimeError."""
        self.model_exp.fit()
        params = (self.model_exp.get_discrimination(), np.log(self.model_exp.get_base_rate() / (1 - self.model_exp.get_base_rate())))
        predictions = self.model_exp.predict(params)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_fit_improves_likelihood(self):
        """Ensure fitting improves likelihood score."""
        init_params = [1.0, 0.0]
        nll_initial = self.model_exp.negative_log_likelihood(init_params)
        self.model_exp.fit()
        fitted_params = [self.model_exp.get_discrimination(),
                         np.log(self.model_exp.get_base_rate() / (1 - self.model_exp.get_base_rate()))]
        nll_fitted = self.model_exp.negative_log_likelihood(fitted_params)
        self.assertLess(nll_fitted, nll_initial)

if __name__ == '__main__':
    unittest.main()
