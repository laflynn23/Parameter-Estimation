import unittest
from src.SimplifiedThreePL import SimplifiedThreePL

class TestSimplifiedThreePL(unittest.TestCase):

    def test_probability_output(self):
        model = SimplifiedThreePL(0.5, 1.2, 0.2)
        result = model.probability(1.0)
        self.assertTrue(0 <= result <= 1, "Probability should be between 0 and 1.")

    def test_invalid_guessing_param(self):
        with self.assertRaises(ValueError):
            SimplifiedThreePL(1.0, 1.0, 1.5)  # Guessing > 1 (invalid)

    def test_invalid_discrimination_param(self):
        with self.assertRaises(ValueError):
            SimplifiedThreePL(1.0, -1.0, 0.2)  # Negative discrimination (invalid)

if __name__ == "__main__":
    unittest.main()
