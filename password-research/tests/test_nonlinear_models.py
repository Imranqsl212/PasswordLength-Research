"""Validate aggregate targets, sufficient statistics, and training-only bases."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/nonlinear_models.py"
SPEC = importlib.util.spec_from_file_location("nonlinear_models", MODULE_PATH)
models = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = models
SPEC.loader.exec_module(models)


def histogram_fixture():
    return pd.DataFrame([
        (6, 100, 1), (6, 10, 2), (6, 1, 7),
        (7, 50, 1), (7, 10, 4), (7, 1, 5),
        (8, 5, 1), (8, 1, 9),
    ], columns=["length", "frequency", "multiplicity"])


def smooth_groups():
    # Synthetic aggregate fixtures only, with both target classes in each group.
    lengths = np.arange(6, 17)
    sizes = np.arange(11, 22, dtype=float) * 100
    q = 1 / (1 + np.exp(1.2 + .18 * (lengths - 6)))
    return pd.DataFrame({"length": lengths, "anonymous_bins": sizes,
                         "fractional_early_bins": sizes * q,
                         "observed_distinct_early_fraction": q})


class NonlinearModelTests(unittest.TestCase):
    def test_fractional_pooled_boundary_and_distinct_weights(self):
        groups, metadata = models.prepare_groups(histogram_fixture(), [6, 7, 8], 5)
        self.assertEqual(metadata["pooled_anonymous_bins"], 30)
        self.assertEqual(metadata["pooled_topk_cutoff_frequency"], 10)
        np.testing.assert_allclose(groups.fractional_early_bins, [2, 3, 0])
        np.testing.assert_allclose(groups.observed_distinct_early_fraction, [.2, .3, 0])
        self.assertEqual(groups.anonymous_bins.sum(), 30)
        self.assertEqual(groups.fractional_early_bins.sum(), 5)

    def test_full_pool_includes_nonmodeled_length_groups(self):
        histogram = pd.concat([histogram_fixture(), pd.DataFrame([(20, 900, 1)],
                            columns=["length", "frequency", "multiplicity"])], ignore_index=True)
        groups, metadata = models.prepare_groups(histogram, [6, 7, 8], 5)
        self.assertEqual(metadata["pooled_anonymous_bins"], 31)
        self.assertEqual(metadata["pooled_length_groups"], 4)
        self.assertAlmostEqual(groups.fractional_early_bins.sum(), 4)
        self.assertAlmostEqual(groups.iloc[0].fractional_early_bins, 1 + 2 / 3)

    def test_sufficient_statistics_preserve_fractional_bernoulli_objective(self):
        groups, _ = models.prepare_groups(histogram_fixture(), [6, 7, 8], 5)
        x, y, weights = models.binary_design(groups)
        self.assertAlmostEqual(weights.sum(), groups.anonymous_bins.sum())
        for row in groups.itertuples():
            selected = x == row.length
            self.assertAlmostEqual(float(np.average(y[selected], weights=weights[selected])),
                                   row.observed_distinct_early_fraction)
            p = .17 + row.length / 100
            expanded_log = float(np.average(-(y[selected] * math.log(p) + (1 - y[selected]) * math.log1p(-p)),
                                            weights=weights[selected]))
            expanded_brier = float(np.average((y[selected] - p) ** 2, weights=weights[selected]))
            scores = models.scoring(row.observed_distinct_early_fraction, p)
            self.assertAlmostEqual(scores["log_loss"], expanded_log)
            self.assertAlmostEqual(scores["brier_loss"], expanded_brier)

    def test_invalid_release_inputs_and_unsupported_budgets_rejected(self):
        for budget in (0, 31, 2.5):
            with self.assertRaises(ValueError):
                models.prepare_groups(histogram_fixture(), [6, 7, 8], budget)
        for value in (-1, np.nan, 2.5):
            histogram = histogram_fixture().astype(float)
            histogram.loc[0, "multiplicity"] = value
            with self.assertRaises(ValueError):
                models.prepare_groups(histogram, [6, 7, 8], 5)
        with self.assertRaises(ValueError):
            models.prepare_groups(histogram_fixture(), [6, 9], 5)

    def test_standardization_uses_training_weights_only(self):
        spec = models.ModelSpec("quadratic", "polynomial", degree=2)
        basis = models.LengthBasis(spec).fit(np.asarray([6., 8.]), np.asarray([1., 3.]))
        self.assertEqual(basis.mean, 7.5)
        self.assertAlmostEqual(basis.sd, math.sqrt(.75))
        before = (basis.mean, basis.sd, basis.min_length, basis.max_length)
        transformed = basis.transform(np.asarray([16., 1000.]))
        self.assertEqual(transformed.shape, (2, 2))
        self.assertEqual(before, (basis.mean, basis.sd, basis.min_length, basis.max_length))
        np.testing.assert_allclose(transformed[:, 1], transformed[:, 0] ** 2)

    def test_spline_training_knots_exclude_held_endpoint(self):
        train = smooth_groups().query("length != 16")
        x, _, weights = models.binary_design(train)
        spec = models.ModelSpec("spline", "bspline", degree=3, n_knots=4)
        basis = models.LengthBasis(spec).fit(x, weights)
        knots = np.asarray(basis.details()["spline_knot_vector_source_units"])
        # Ignore the three added boundary knots on each side.
        np.testing.assert_allclose(knots[3:-3], [6, 9, 12, 15])
        before = knots.copy()
        result = basis.transform(np.asarray([16., 17.]))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result.shape[1], 5)
        np.testing.assert_allclose(before, basis.details()["spline_knot_vector_source_units"])
        # This extrapolation has fixed first derivative outside training support.
        np.testing.assert_allclose(basis.transform(np.asarray([17., 18.]))[1] - result[1],
                                   result[1] - result[0], atol=1e-12)

    def test_intercept_is_exact_training_fraction(self):
        groups = smooth_groups()
        fit = models.fit_model(models.MODEL_SPECS[0], groups)
        expected = groups.fractional_early_bins.sum() / groups.anonymous_bins.sum()
        np.testing.assert_allclose(fit.predict(np.asarray([2, 16, 100])), expected)
        self.assertEqual(fit.iterations, 0)

    def test_full_and_held_calibration_folds_and_counts(self):
        groups = smooth_groups()
        results, validation = models.evaluate(groups)
        self.assertEqual(len(results), 6)
        self.assertEqual(len(validation), 6 * 11 * 2)
        self.assertFalse(validation.convergence_warning.astype(bool).any())
        for name, data in validation.groupby("model"):
            held = data[data.fit_scope == "held_length"]
            full = data[data.fit_scope == "full_data"]
            self.assertEqual(held.held_length.tolist(), list(range(6, 17)))
            self.assertEqual(full.held_length.tolist(), list(range(6, 17)))
            np.testing.assert_allclose(held.training_anonymous_bins + held.test_anonymous_bins,
                                       groups.anonymous_bins.sum())
            self.assertEqual(int(held.query("held_length == 16").training_max_length.iloc[0]), 15)
            summary = results[results.model == name].iloc[0]
            self.assertAlmostEqual(summary.held_length_log_loss,
                                   np.average(held.log_loss, weights=held.test_anonymous_bins))
            self.assertTrue(((data.predicted_fraction > 0) & (data.predicted_fraction < 1)).all())

    def test_log_loss_handles_extremes_and_brier_is_not_calibration_square(self):
        for observed, predicted in ((0, 0), (1, 1), (.5, 0), (.5, 1)):
            scores = models.scoring(observed, predicted)
            self.assertTrue(all(np.isfinite(value) for value in scores.values()))
        scores = models.scoring(.2, .2)
        self.assertAlmostEqual(scores["squared_calibration_error"], 0)
        self.assertAlmostEqual(scores["brier_loss"], .16)
        with self.assertRaises(ValueError):
            models.scoring(.2, 1.1)

    def test_linear_agreement_check_rejects_changed_predictions_or_targets(self):
        _, validation = models.evaluate(smooth_groups(), [models.MODEL_SPECS[1]])
        legacy = validation[validation.fit_scope == "held_length"][[
            "held_length", "predicted_fraction", "observed_distinct_early_fraction", "log_loss"]].copy()
        self.assertEqual(models.compare_legacy(validation, legacy)["status"], "PASSED")
        changed = legacy.copy()
        changed.loc[changed.index[0], "predicted_fraction"] *= 2
        with self.assertRaises(AssertionError):
            models.compare_legacy(validation, changed)
        changed = legacy.copy()
        changed.loc[changed.index[0], "observed_distinct_early_fraction"] = 0
        with self.assertRaises(AssertionError):
            models.compare_legacy(validation, changed)


if __name__ == "__main__":
    unittest.main()
