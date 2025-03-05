import numpy as np
import math
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment=None, difficulty=None, discrimination=1.0, base_rate=0.2):
        """
        Initialize either:
        - A single-item model (`difficulty`, `discrimination`, `base_rate`) OR
        - An experiment-based model (`experiment` object with multiple conditions).
        """
        self.experiment = experiment
        self._is_fitted = False

        if experiment is not None:
            # Multi-condition mode (Experiment object provided)
            self._difficulties = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
            if len(self.experiment.conditions) != len(self._difficulties):
                raise ValueError("Number of conditions in experiment does not match expected difficulties")
            self._discrimination = None
            self._logit_base_rate = None
            self._base_rate = None
        else:
            # Single difficulty mode
            self.difficulty = difficulty
            self.discrimination = discrimination
            self.base_rate = base_rate
            self._validate_inputs()

    def _validate_inputs(self):
        """Ensure single-mode parameters are valid."""
        if not (0 <= self.base_rate <= 1):
            raise ValueError("Base rate must be between 0 and 1.")
        if self.discrimination <= 0:
            raise ValueError("Discrimination parameter must be positive.")

    def summary(self):
        """Return summary statistics if an experiment is used."""
        if self.experiment is None:
            raise RuntimeError("Summary only available when using an experiment.")
        n_correct = sum(sdt.n_correct_responses() for sdt in self.experiment.conditions)
        n_incorrect = sum(sdt.n_incorrect_responses() for sdt in self.experiment.conditions)
        return {
            "n_total": n_correct + n_incorrect,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": len(self._difficulties)
        }

    def probability(self, theta):
        """
        Compute probability for a single-item model.
        """
        exponent = self.discrimination * (theta - self.difficulty)
        return self.base_rate + (1 - self.base_rate) / (1 + math.exp(-exponent))

    def predict(self, parameters):
        """
        Compute probability for experiment-based models.
        """
        a, c_logit = parameters
        c = 1.0 / (1.0 + np.exp(-c_logit))

        if self.experiment is not None:
            p = c + (1 - c) * (1.0 / (1.0 + np.exp(a * self._difficulties)))
            return p
        else:
            raise RuntimeError("Predict is only for experiment-based models.")

    def negative_log_likelihood(self, parameters):
        """
        Compute the negative log-likelihood.
        """
        if self.experiment is None:
            raise RuntimeError("Likelihood calculation only for experiment-based models.")

        if not self._is_fitted:
            a, c_logit = parameters
            c = 1.0 / (1.0 + np.exp(-c_logit))
            p = c + (1 - c) * (1.0 / (1.0 + np.exp(a * self._difficulties)))
        else:
            p = self.predict(parameters)

        p = np.clip(p, 1e-9, 1 - 1e-9)

        nll = sum(-sdt.n_correct_responses() * np.log(p[i]) - sdt.n_incorrect_responses() * np.log(1 - p[i])
                  for i, sdt in enumerate(self.experiment.conditions))
        return nll

    def fit(self):
        """Fit the model to experimental data."""
        if self.experiment is None:
            raise RuntimeError("Fitting only available for experiment mode.")

        result = minimize(self.negative_log_likelihood, [1.0, 0.0], method='Nelder-Mead')

        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        self._discrimination, self._logit_base_rate = result.x
        self._base_rate = 1.0 / (1.0 + np.exp(-self._logit_base_rate))
        self._is_fitted = True

    def get_discrimination(self):
        """Return the discrimination parameter."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        """Return the base rate parameter."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return self._base_rate
