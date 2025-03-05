import unittest
import numpy as np
from SimplifiedThreePL import SimplifiedThreePL
from Experiment import Experiment
from SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        # Create an experiment with 5 conditions using dummy SignalDetection objects.
        # Here each condition has 25 hits, 25 misses, 25 false alarms, and 25 correct rejections.
        self.experiment = Experiment()
        for _ in range(5):
            sdt = SignalDetection(25, 25, 25, 25)
            self.experiment.add_condition(sdt)
        self.model = SimplifiedThreePL(self.experiment)

    def test_summary(self):
        summary = self.model.summary()
        # Each condition: correct = 25+25=50, incorrect = 25+25=50; total = 100 per condition, 500 overall.
        self.assertEqual(summary["n_total"], 500)
        self.assertEqual(summary["n_correct"], 250)
        self.assertEqual(summary["n_incorrect"], 250)
        self.assertEqual(summary["n_conditions"], 5)

    def test_invalid_initialization(self):
        # Test that providing an experiment with the wrong number of conditions raises an error.
        exp_invalid = Experiment()
        # Only add 4 conditions instead of 5.
        for _ in range(4):
            sdt = SignalDetection(25, 25, 25, 25)
            exp_invalid.add_condition(sdt)
        with self.assertRaises(ValueError):
            _ = SimplifiedThreePL(exp_invalid)

    def test_predict_range(self):
        # Test that predict() outputs probabilities between 0 and 1.
        params = (1.0, 0.0)  # a = 1, c_logit = 0 => base rate = 0.5
        predictions = self.model.predict(params)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_predict_effects(self):
        # Higher base rate should increase predicted probabilities.
        params_low = (1.0, -2.0)   # lower c
        params_high = (1.0, 2.0)     # higher c
        pred_low = self.model.predict(params_low)
        pred_high = self.model.predict(params_high)
        self.assertTrue(np.all(pred_high >= pred_low))
        
        # With positive discrimination, higher difficulty should yield lower probability.
        params = (1.0, 0.0)
        predictions = self.model.predict(params)
        # Difficulties are [2, 1, 0, -1, -2]; condition with difficulty 2 should have a lower probability
        # than condition with difficulty -2.
        self.assertLess(predictions[0], predictions[-1])
        
    def test_get_parameters_before_fit(self):
        # The getters should raise an error if the model hasn’t been fitted.
        with self.assertRaises(RuntimeError):
            _ = self.model.get_discrimination()
        with self.assertRaises(RuntimeError):
            _ = self.model.get_base_rate()

    def test_fit_improves_likelihood(self):
        # Check that the negative log-likelihood improves after fitting.
        init_params = [1.0, 0.0]
        nll_initial = self.model.negative_log_likelihood(init_params)
        self.model.fit()
        fitted_params = [self.model.get_discrimination(),
                         np.log(self.model.get_base_rate() / (1 - self.model.get_base_rate()))]
        nll_fitted = self.model.negative_log_likelihood(fitted_params)
        self.assertLess(nll_fitted, nll_initial)

    def test_integration(self):
        # Integration test with a dataset having 100 trials per condition
        # with accuracies: 0.55, 0.60, 0.75, 0.90, and 0.95.
        # For each condition, we split 100 trials equally between signal (50) and noise (50).
        # We manually set counts so that:
        #   n_correct = (hits + correctRejections) equals the desired number,
        #   n_incorrect = (misses + falseAlarms) equals 100 minus that.
        experiment = Experiment()
        # Condition 1: 55 correct; set hits=28, misses=22, correctRejections=27, falseAlarms=23.
        experiment.add_condition(SignalDetection(28, 22, 23, 27))
        # Condition 2: 60 correct; hits=30, misses=20, correctRejections=30, falseAlarms=20.
        experiment.add_condition(SignalDetection(30, 20, 20, 30))
        # Condition 3: 75 correct; hits=38, misses=12, correctRejections=37, falseAlarms=13.
        experiment.add_condition(SignalDetection(38, 12, 13, 37))
        # Condition 4: 90 correct; hits=45, misses=5, correctRejections=45, falseAlarms=5.
        experiment.add_condition(SignalDetection(45, 5, 5, 45))
        # Condition 5: 95 correct; hits=48, misses=2, correctRejections=47, falseAlarms=3.
        experiment.add_condition(SignalDetection(48, 2, 3, 47))
        
        model = SimplifiedThreePL(experiment)
        model.fit()
        # Use the fitted parameters to generate predictions.
        predictions = model.predict((model.get_discrimination(), model._logit_base_rate))
        # Observed accuracies (correct/total) for each condition.
        observed = [55/100, 60/100, 75/100, 90/100, 95/100]
        # Allow a small tolerance for matching.
        for pred, obs in zip(predictions, observed):
            self.assertAlmostEqual(pred, obs, delta=0.1)

    def test_stability_of_fit(self):
        # Fit the model twice and ensure the parameters are stable.
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