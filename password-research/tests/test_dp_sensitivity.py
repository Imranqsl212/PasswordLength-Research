import math
import unittest
from collections import Counter

from experiments.dp_sensitivity import (
    fraction_envelope, illustrative_radius, log_tail_bound,
    metric_counts, required_radius, spectrum_l1,
)


def partitions(n, maximum=None):
    if not n:
        yield ()
        return
    for first in range(min(n, maximum or n), 0, -1):
        for rest in partitions(n - first, first):
            yield (first,) + rest


def histogram(vector):
    return sorted(Counter(vector).items(), reverse=True)


class DPSensitivityTests(unittest.TestCase):
    def test_l1_matches_expansion_and_zero_padding(self):
        cases = [([], []), ([(2, 3)], []), ([(4, 2), (1, 3)], [(3, 1), (2, 4)]),
                 ([(2, 1000000)], [(1, 1000001)])]
        expected = [0, 6, 6, 1000001]
        for (first, second), value in zip(cases, expected):
            self.assertEqual(spectrum_l1(first, second), value)
            self.assertEqual(spectrum_l1(second, first), value)

    def test_small_partitions_count_indicators_and_envelopes(self):
        vectors = [p for n in range(1, 8) for p in partitions(n)]
        for first in vectors:
            h1 = histogram(first)
            counts1 = metric_counts(h1)
            for second in vectors:
                h2 = histogram(second)
                radius = spectrum_l1(h1, h2)
                counts2 = metric_counts(h2)
                self.assertLessEqual(abs(len(first) - len(second)), radius)
                for key, count in counts1.items():
                    self.assertLessEqual(abs(count - counts2[key]), radius)
                    lo, hi = fraction_envelope(count, len(first), radius)
                    actual = counts2[key] / len(second)
                    self.assertLessEqual(lo, actual + 1e-12)
                    self.assertLessEqual(actual, hi + 1e-12)

    def test_required_radius_is_necessary(self):
        first = (2, 2, 2, 2)
        count, denominator = 4, 4
        for n in range(1, 10):
            for second in partitions(n):
                target = second.count(2) / len(second)
                necessary = required_radius(count, denominator, target)
                self.assertLessEqual(necessary, spectrum_l1(histogram(first), histogram(second)))

    def test_tail_radius_and_parameter_guards(self):
        for assumed in (1000, 15212723, 30425446, 174292189):
            radius = illustrative_radius(assumed)
            self.assertLessEqual(log_tail_bound(assumed, radius), math.log(0.05) + 1e-12)
            self.assertGreater(log_tail_bound(assumed, radius - 1), math.log(0.05))
        self.assertEqual(illustrative_radius(174292189), 271173)
        self.assertIsNone(illustrative_radius(100))
        self.assertIsNone(log_tail_bound(100, 1000))
        with self.assertRaises(ValueError):
            illustrative_radius(1000, beta=0)

    def test_bounds_are_vacuous_at_denominator_and_exact_at_zero(self):
        self.assertEqual(fraction_envelope(2, 5, 5), (0.0, 1.0))
        self.assertEqual(fraction_envelope(2, 5, 0), (0.4, 0.4))
        with self.assertRaises(ValueError):
            fraction_envelope(1, 0, 1)

    def test_count_indicators_ignore_explicit_zero_padding(self):
        first = [(4, 1), (2, 2), (1, 3)]
        self.assertEqual(metric_counts(first), metric_counts(first + [(0, 1000)]))


if __name__ == "__main__":
    unittest.main()
