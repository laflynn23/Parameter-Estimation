import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # logistic (sigmoid) function

class SimplifiedThreePL:
    def __init__(self, experiment):
        """
        Initialize the model with an Experiment object.
        The Experiment should have a list of SignalDetection objects in its 'conditions' attribute.
        """
        self.experiment = experiment
        self._difficulties = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        if len(self.experiment.conditions) != len(self._difficulties):
            raise ValueError("Number of conditions in experiment does not match default difficulties")
        self._discrimination = None
        self._logit_base_rate = None  # internal parameter for base rate
        self._base_rate = None
        self._is_fitted = False

    def summary(self):
        """Return a dictionary with summary statistics based on the SignalDetection conditions."""
        n_correct = sum(sdt.n_correct_responses() for sdt in self.experiment.conditions)
        n_incorrect = sum(sdt.n_incorrect_responses() for sdt in self.experiment.conditions)
        n_total = n_correct + n_incorrect
        n_conditions = len(self._difficulties)
        return {
            "n_total": int(n_total),
            "n_correct": int(n_correct),
            "n_incorrect": int(n_incorrect),
            "n_conditions": n_conditions
        }

    def predict(self, parameters):
        """
        Given parameters (a, c_logit), compute the probability of a correct response in each condition.
        a: discrimination parameter
        c_logit: the logit of the base rate parameter (so that c = 1/(1+exp(-c_logit)))
        
        Returns:
            numpy array of probabilities for each condition.
        """
        a, c_logit = parameters
        c = 1.0 / (1.0 + np.exp(-c_logit))
        # The probability function is defined as:
        # p = c + (1-c) * logistic( a * (theta - b) )
        # where theta is fixed at 0 and b are the difficulties.
        # This simplifies to:
        # p = c + (1-c) * (1/(1+exp(a * b)))
        p = c + (1 - c) * (1.0 / (1.0 + np.exp(a * self._difficulties)))
        return p

    def negative_log_likelihood(self, parameters):
        """
        Compute the negative log-likelihood of the observed data given the parameters.
        For each condition, the observed counts are obtained from the SignalDetection object.
        """
        p = self.predict(parameters)
        # Avoid log(0) by clipping probabilities.
        eps = 1e-9
        p = np.clip(p, eps, 1 - eps)
        nll = 0.0
        for i, sdt in enumerate(self.experiment.conditions):
            correct = sdt.n_correct_responses()
            errors = sdt.n_incorrect_responses()
            nll -= correct * np.log(p[i]) + errors * np.log(1 - p[i])
        return nll

    def fit(self):
        """
        Fit the model by minimizing the negative log likelihood.
        Stores the fitted discrimination and base rate parameters.
        """
        # Initial guess: a = 1.0, c_logit = 0.0 (so base rate c = 0.5)
        init_params = [1.0, 0.0]
        result = minimize(self.negative_log_likelihood, init_params, method='Nelder-Mead')
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)
        a_fit, c_logit_fit = result.x
        self._discrimination = a_fit
        self._logit_base_rate = c_logit_fit
        self._base_rate = 1.0 / (1.0 + np.exp(-c_logit_fit))
        self._is_fitted = True

    def get_discrimination(self):
        """Return the discrimination parameter a. Raises an error if the model isn’t fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
        return self._discrimination

    def get_base_rate(self):
        """Return the base rate parameter c (not the logit). Raises an error if the model isn’t fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")
        return self._base_rate