import math
import unittest
from prepare_cpu import trailing_zero_count, quantile_linear, eligible, prior_delta, onehot


class PreparationTests(unittest.TestCase):
    def test_suffix(self):
        self.assertEqual(trailing_zero_count([0, 4, 0, 0]), 2)
        self.assertEqual(trailing_zero_count([0, 0, 4]), 0)
        self.assertEqual(trailing_zero_count([0, 0]), 2)
        self.assertEqual(trailing_zero_count([]), 0)

    def test_quantile_includes_zeros(self):
        self.assertEqual(quantile_linear([0, 0, 10, 30], .5), 5)
        self.assertAlmostEqual(quantile_linear([0, 0, 10, 30], .9), 24)

    def test_duration_boundary(self):
        self.assertTrue(eligible(10 * 16000, 16000, 2))
        self.assertFalse(eligible(10 * 16000 + 1, 16000, 2))
        self.assertTrue(eligible(11 * 16000, 16000, 1))
        self.assertFalse(eligible(11 * 16000, 16000, 2))

    def test_log_ratio(self):
        self.assertAlmostEqual(prior_delta({'neutral': 10, 'happy': 20}, {'neutral': 10, 'happy': 10}, 'happy'), math.log(2))
        self.assertIsNone(prior_delta({'neutral': 10, 'happy': 0}, {'neutral': 10, 'happy': 1}, 'happy'))

    def test_onehot(self):
        self.assertEqual(onehot(dict(Excited='0', Unconfident='1', Neutral_3Class='0')), 'Unconfident')
        with self.assertRaises(ValueError):
            onehot(dict(Excited='.5', Unconfident='.5', Neutral_3Class='0'))


if __name__ == '__main__':
    unittest.main()
