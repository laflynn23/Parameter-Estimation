import numpy as np
from scipy.optimize import minimize

class SimplifiedThreePL:
    def __init__(self, experiment):
        # The experiment object contains the data for correct and incorrect responses.
        self.experiment = experiment
        self._difficulties = np.array([2.0, 1.0, 0.0, -1.0, -2.0])  # predefined difficulty levels
        self._ability = 0.0  # fixed participant ability
        self._base_rate = None  # base rate parameter (to be estimated)
        self._logit_base_rate = None  # logit transformation of the base rate
        self._discrimination = None  # discrimination parameter (to be estimated)
        self._is_fitted = False  # whether the model is fitted with parameters

    def summary(self):
        # Returns a dictionary with key statistics from the experiment
        correct_responses = self.experiment.correct_responses
        total_responses = self.experiment.total_responses
        n_conditions = len(self._difficulties)

        return {
            "n_total": total_responses,
            "n_correct": correct_responses,
            "n_incorrect": total_responses - correct_responses,
            "n_conditions": n_conditions
        }

    def predict(self, parameters):
        """ Predict the probability of correct responses for each condition given the parameters. """
        discrimination, logit_base_rate = parameters
        base_rate = 1 / (1 + np.exp(-logit_base_rate))  # inverse logit transformation
        
        # Calculate the probability of a correct response for each condition
        probabilities = base_rate + (1 - base_rate) / (1 + np.exp(-discrimination * (self._difficulties - self._ability)))
        return probabilities

    def negative_log_likelihood(self, parameters):
        """ Compute the negative log-likelihood function for the data given the parameters. """
        discrimination, logit_base_rate = parameters
        base_rate = 1 / (1 + np.exp(-logit_base_rate))  # inverse logit transformation
        probabilities = self.predict(parameters)
        
        correct_responses = self.experiment.correct_responses
        total_responses = self.experiment.total_responses
        incorrect_responses = total_responses - correct_responses
        
        log_likelihood = np.sum(correct_responses * np.log(probabilities) + incorrect_responses * np.log(1 - probabilities))
        return -log_likelihood  # negative log-likelihood

    def fit(self):
        """ Fit the model using maximum likelihood estimation (MLE). """
        initial_guess = [1.0, 0.0]  # initial guess for discrimination and logit base rate
        result = minimize(self.negative_log_likelihood, initial_guess, method="BFGS")

        # After fitting, store the estimated parameters
        self._discrimination, self._logit_base_rate = result.x
        self._base_rate = 1 / (1 + np.exp(-self._logit_base_rate))  # inverse logit
        self._is_fitted = True

    def get_discrimination(self):
        """ Return the estimated discrimination parameter, if the model is fitted. """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet!")
        return self._discrimination

    def get_base_rate(self):
        """ Return the estimated base rate parameter, if the model is fitted. """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet!")
        return self._base_rate
