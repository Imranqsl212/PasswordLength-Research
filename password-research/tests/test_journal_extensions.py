import math
import unittest
import numpy as np

from experiments.journal_extensions import (
    groups_for_threshold, held_threshold, group_exposure,
    threshold_from_histograms, cross_dataset_concentration,
)
from experiments.nonlinear_models import scoring
from src.analysis import merge_histograms


class JournalExtensionTests(unittest.TestCase):
    def test_held_histogram_cannot_change_training_threshold(self):
        original = {6: [(5, 1), (2, 2), (1, 1)], 8: [(4, 1), (2, 3)], 10: [(100, 100)]}
        changed = {**original, 10: [(1, 10000000)]}
        first = held_threshold(original, [6, 8, 10], 10, 4)
        second = held_threshold(changed, [6, 8, 10], 10, 4)
        self.assertEqual(first, second)
        self.assertEqual(first.cutoff_frequency, 2)
        self.assertEqual(first.boundary_fraction, 2 / 5)

    def test_frozen_rule_applies_to_unseen_frequency_values(self):
        threshold = threshold_from_histograms([[(5, 1), (2, 4), (1, 3)]], 3)
        self.assertEqual(threshold.fraction(4), 1)
        self.assertEqual(threshold.fraction(2), .5)
        self.assertEqual(threshold.fraction(1), 0)
        data = {6: [(5, 1), (2, 4), (1, 3)], 10: [(4, 2), (2, 2), (1, 6)]}
        train = groups_for_threshold(data, [6], threshold).iloc[0]
        held = groups_for_threshold(data, [10], threshold).iloc[0]
        self.assertEqual(train.fractional_early_bins, 3)
        self.assertEqual(held.fractional_early_bins, 3)
        self.assertEqual(held.observed_distinct_early_fraction, .3)

    def test_binary_fractional_scoring_matches_explicit_weighted_labels(self):
        weights, labels, p = np.array([2., 8.]), np.array([1., 0.]), .3
        got = scoring(.2, p)
        expected_log = np.average(-(labels * math.log(p) + (1 - labels) * math.log(1 - p)), weights=weights)
        expected_brier = np.average((labels - p) ** 2, weights=weights)
        self.assertAlmostEqual(got["log_loss"], expected_log)
        self.assertAlmostEqual(got["brier_loss"], expected_brier)
        self.assertNotAlmostEqual(got["brier_loss"], (.2 - p) ** 2)

    def test_long_tie_expectation_and_sharp_bounds(self):
        group = [(2, 2)]
        pool = [(4, 1), (2, 4), (1, 2)]
        got = group_exposure(group, pool, 2)
        self.assertEqual(got["account_fraction"], .25)
        self.assertEqual(got["account_tie_lower"], 0)
        self.assertEqual(got["account_tie_upper"], .5)
        self.assertEqual(got["distinct_fraction"], .25)
        self.assertEqual(got["distinct_tie_lower"], 0)
        self.assertEqual(got["distinct_tie_upper"], .5)
        self.assertEqual(group_exposure(group, pool, 1)["expected_selected_bins"], 0)
        all_bins = group_exposure(pool, pool, 2)
        self.assertEqual(all_bins["expected_selected_account_mass"], 6)
        self.assertEqual(all_bins["expected_selected_bins"], 2)

    def test_cross_dataset_denominators_and_budget_guard(self):
        result = cross_dataset_concentration([(4, 1), (1, 2)], [(2, 2)], [1]).iloc[0]
        self.assertAlmostEqual(result.linkedin_full_coverage, 4 / 6)
        self.assertAlmostEqual(result.yahoo_all_coverage, .5)
        self.assertAlmostEqual(result.yahoo_minus_linkedin_percentage_points, 100 * (.5 - 4 / 6))
        with self.assertRaises(ValueError):
            threshold_from_histograms([[(1, 2)]], 3)


if __name__ == "__main__":
    unittest.main()
