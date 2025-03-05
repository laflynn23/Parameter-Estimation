import math

class SimplifiedThreePL:
    def __init__(self, difficulty: float, discrimination: float, guessing: float):
        self.difficulty = difficulty
        self.discrimination = discrimination
        self.guessing = guessing
        
        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure parameters are valid."""
        if not (0 <= self.guessing <= 1):
            raise ValueError("Guessing parameter must be between 0 and 1.")
        if self.discrimination <= 0:
            raise ValueError("Discrimination parameter must be positive.")

    def probability(self, theta: float) -> float:
        """Compute the probability of a correct response given ability (theta)."""
        exponent = self.discrimination * (theta - self.difficulty)
        return self.guessing + (1 - self.guessing) / (1 + math.exp(-exponent))
