import unittest
from analyze_n3_ab_predictions_cpu import bucket, metrics


class DiagnosticsTests(unittest.TestCase):
    def test_duration_boundaries(self):
        for duration, expected in ((.5, '<1'), (1, '1-2'), (2, '2-4'),
                                   (4, '4-8'), (8, '8-12'), (12, '8-12')):
            self.assertEqual(bucket(duration), expected)

    def test_invalid_duration(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(RuntimeError):
                bucket(value)

    def test_perfect(self):
        r = metrics([0, 1, 2], [0, 1, 2])
        self.assertEqual(r['macro_f1'], 1)
        self.assertEqual(r['accuracy'], 1)

    def test_empty_and_absent_class(self):
        for truth, pred in (([], []), ([0, 2], [0, 1])):
            r = metrics(truth, pred)
            self.assertIsNone(r['macro_f1'])
            self.assertIsNone(r['per_class']['Unconfident']['recall'])

    def test_zero_recall_not_missing(self):
        r = metrics([0, 1, 2], [0, 2, 2])
        self.assertEqual(r['per_class']['Unconfident']['recall'], 0)
        self.assertIsNone(r['per_class']['Unconfident']['precision'])

    def test_label_and_length_validation(self):
        for truth, pred in (([0], []), ([3], [0])):
            with self.assertRaises(RuntimeError):
                metrics(truth, pred)


if __name__ == '__main__':
    unittest.main()
