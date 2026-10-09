"""Fixed exploratory journal extensions on anonymous released aggregates.

Preserves original code/tables. No strings, guessers, raw credentials,
prospective-user inference, or population uncertainty estimates.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nonlinear_models import MODEL_SPECS, fit_model, scoring
from src.analysis import read_sources, merge_histograms
from src.core import coverage, mass, size, pooled_tie_coverage, rank_map


@dataclass(frozen=True)
class Threshold:
    cutoff_frequency: int
    boundary_multiplicity: int
    boundary_selected_bins: int
    bins_strictly_above: int
    training_anonymous_bins: int
    budget: int

    @property
    def boundary_fraction(self):
        return self.boundary_selected_bins / self.boundary_multiplicity

    def fraction(self, frequency):
        if frequency > self.cutoff_frequency:
            return 1.0
        if frequency < self.cutoff_frequency:
            return 0.0
        return self.boundary_fraction


def threshold_from_histograms(histograms, budget):
    pool = merge_histograms(histograms)
    total = size(pool)
    if not isinstance(budget, int) or not 1 <= budget <= total:
        raise ValueError("Budget must be within the training positive-bin count")
    before = 0
    for frequency, multiplicity in pool:
        if before + multiplicity >= budget:
            return Threshold(frequency, multiplicity, budget - before, before, total, budget)
        before += multiplicity
    raise AssertionError("No training cutoff found")


def held_threshold(lengths, primary, held, budget):
    if held not in primary or len(set(primary)) != len(primary):
        raise ValueError("Held length must belong to unique primary groups")
    missing = set(primary) - set(lengths)
    if missing:
        raise ValueError(f"Missing primary groups: {sorted(missing)}")
    # Only the other groups' numerical histograms reach threshold construction.
    return threshold_from_histograms([lengths[length] for length in primary if length != held], budget)


def groups_for_threshold(lengths, selected, threshold):
    rows = []
    for length in sorted(selected):
        histogram = lengths[length]
        count = size(histogram)
        positive = math.fsum(m * threshold.fraction(f) for f, m in histogram)
        rows.append({"length": length, "anonymous_bins": count,
                     "fractional_early_bins": positive,
                     "observed_distinct_early_fraction": positive / count})
    return pd.DataFrame(rows)


def training_threshold_models(lengths, plan):
    primary = plan["primary_lengths"]
    budget = plan["training_threshold_budget"]
    models = {spec.name: spec for spec in MODEL_SPECS}
    if set(plan["models"]) != set(models):
        raise ValueError("Plan must retain all six existing fixed model specifications")
    validation, thresholds = [], []
    for held in primary:
        threshold = held_threshold(lengths, primary, held, budget)
        training_lengths = [length for length in primary if length != held]
        training = groups_for_threshold(lengths, training_lengths, threshold)
        test = groups_for_threshold(lengths, [held], threshold).iloc[0]
        positive = float(training.fractional_early_bins.sum())
        if not math.isclose(positive, budget, abs_tol=1e-8, rel_tol=0):
            raise AssertionError("Training threshold does not select the declared bin budget")
        baseline = positive / threshold.training_anonymous_bins
        cutoff_row = {"held_length": held, **asdict(threshold),
                      "boundary_fraction": threshold.boundary_fraction,
                      "training_fractional_positive_bins": positive,
                      "training_constant_fraction": baseline,
                      "held_anonymous_bins": int(test.anonymous_bins),
                      "held_fractional_positive_bins": float(test.fractional_early_bins),
                      "held_observed_fraction": float(test.observed_distinct_early_fraction),
                      "status": "training-only frequency-threshold membership; not operational candidate coverage"}
        thresholds.append(cutoff_row)
        for name in plan["models"]:
            model = fit_model(models[name], training)
            predicted = float(model.predict(np.asarray([held], dtype=float))[0])
            observed = float(test.observed_distinct_early_fraction)
            baseline_scores = scoring(observed, baseline)
            validation.append({"model": name, "held_length": held,
                               "cutoff_frequency": threshold.cutoff_frequency,
                               "boundary_fraction": threshold.boundary_fraction,
                               "observed_distinct_fraction": observed, "predicted_fraction": predicted,
                               "test_anonymous_bins": int(test.anonymous_bins),
                               "held_fractional_positive_bins": float(test.fractional_early_bins),
                               "training_anonymous_bins": threshold.training_anonymous_bins,
                               "training_constant_fraction": baseline,
                               **scoring(observed, predicted),
                               **{"constant_" + key: value for key, value in baseline_scores.items()},
                               "training_min_length": min(training_lengths),
                               "training_max_length": max(training_lengths),
                               "is_extrapolation": held < min(training_lengths) or held > max(training_lengths),
                               "iterations": model.iterations, "convergence_warning": model.convergence_warning,
                               "fitted_basis_details": json.dumps(model.parameters(), sort_keys=True),
                               "target_scope": "other ten primary groups; frozen rule applied to held frequency spectrum"})
    validation = pd.DataFrame(validation)
    results = []
    for name in plan["models"]:
        rows = validation[validation.model == name]
        result = {"model": name, "folds": len(rows), "target_weighting": "distinct anonymous bins",
                  "target": "training-only released-frequency threshold membership",
                  "structural_model": "DATA UNAVAILABLE",
                  "validation_scope": "descriptive leave-length-out; no independent users/source or operational guesses",
                  "basis_definition": json.dumps(models[name].definition(), sort_keys=True),
                  "convergence_warning_count": int(rows.convergence_warning.astype(bool).sum()),
                  "max_iterations_used": int(rows.iterations.max())}
        for metric in ("log_loss", "brier_loss", "abs_calibration_error", "squared_calibration_error"):
            result["held_" + metric] = float(np.average(rows[metric], weights=rows.test_anonymous_bins))
            result["constant_" + metric] = float(np.average(rows["constant_" + metric], weights=rows.test_anonymous_bins))
        result["equal_group_log_loss"] = float(rows.log_loss.mean())
        result["log_loss_relative_change_from_fold_constant"] = result["held_log_loss"] / result["constant_log_loss"] - 1
        for length in (15, 16):
            row = rows[rows.held_length == length].iloc[0]
            result[f"length{length}_observed_to_predicted_ratio"] = row.observed_to_predicted_ratio
        results.append(result)
    return pd.DataFrame(results), validation, pd.DataFrame(thresholds)


def group_exposure(group, pool, budget):
    """Both account/bin expected coverage and sharp boundary-tie allocation bounds."""
    account, lower, upper, distinct = pooled_tie_coverage(group, pool, budget)
    positions = rank_map(pool)
    frequencies = dict(pool)
    low_bins = high_bins = 0
    for frequency, multiplicity in group:
        start, _, _ = positions[frequency]
        slots = max(0, min(frequencies[frequency], budget - start + 1))
        low_bins += max(0, slots - (frequencies[frequency] - multiplicity))
        high_bins += min(multiplicity, slots)
    return {"account_fraction": account, "account_tie_lower": lower, "account_tie_upper": upper,
            "expected_selected_account_mass": account * mass(group),
            "distinct_fraction": distinct, "distinct_tie_lower": low_bins / size(group),
            "distinct_tie_upper": high_bins / size(group),
            "expected_selected_bins": distinct * size(group)}


def long_threshold_grid(lengths, plan):
    pool = merge_histograms(lengths.values())
    rows = []
    for minimum in plan["long_minima"]:
        eligible = [histogram for length, histogram in lengths.items() if length >= minimum]
        group = merge_histograms(eligible)
        if not group:
            raise ValueError(f"No supported supplied lengths above {minimum}")
        for budget in plan["coverage_budgets"]:
            if budget > size(pool):
                raise ValueError("Unsupported pooled budget")
            rows.append({"minimum_source_length": minimum, "budget": budget,
                         "released_account_mass": mass(group), "anonymous_bins": size(group),
                         "eligible_length_groups": len(eligible), **group_exposure(group, pool, budget),
                         "original_threshold": minimum == plan["original_long_threshold"]["minimum_length"]
                         and budget == plan["original_long_threshold"]["budget"],
                         "status": "exploratory threshold grid; identityless pooled oracle and released counts"})
    return pd.DataFrame(rows)


def cross_dataset_concentration(full, yahoo, budgets):
    rows = []
    for budget in budgets:
        if budget > min(size(full), size(yahoo)):
            raise ValueError("Unsupported common whole-spectrum budget")
        linked, independent = coverage(full, budget), coverage(yahoo, budget)
        rows.append({"budget": budget, "linkedin_full_coverage": linked, "yahoo_all_coverage": independent,
                     "yahoo_minus_linkedin_percentage_points": 100 * (independent - linked),
                     "linkedin_mass": mass(full), "yahoo_mass": mass(yahoo),
                     "linkedin_bins": size(full), "yahoo_bins": size(yahoo),
                     "status": "separate historical populations; overall concentration only, not length validation or temporal trend"})
    return pd.DataFrame(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root=ROOT):
    root = Path(root).resolve()
    if root.resolve() != ROOT.resolve():
        raise ValueError("Audited source reader is bound to this project root")
    plan_path = root / "analysis/journal-analysis-plan.json"
    plan = json.loads(plan_path.read_text())
    original_tables = {str(path.relative_to(root)): sha(path) for path in (root / "results/tables").glob("*.csv")
                       if not path.name.startswith("journal_")}
    lengths, full, yahoo = read_sources()
    results, validation, thresholds = training_threshold_models(lengths, plan)
    long = long_threshold_grid(lengths, plan)
    cross = cross_dataset_concentration(full, yahoo, plan["coverage_budgets"])
    outputs = {"journal_train_threshold_results.csv": results,
               "journal_train_threshold_validation.csv": validation,
               "journal_train_threshold_targets.csv": thresholds,
               "journal_long_early.csv": long, "journal_cross_dataset.csv": cross}
    for filename, table in outputs.items():
        table.to_csv(root / "results/tables" / filename, index=False)
    after = {name: sha(root / name) for name in original_tables}
    if after != original_tables:
        raise AssertionError("A preserved original table changed")
    sources = [Path(__file__), root / "src/analysis.py", root / "src/core.py", root / "experiments/nonlinear_models.py"]
    metadata = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "COMPLETED_EXPLORATORY_ANONYMOUS_AGGREGATE_EXTENSIONS",
                "plan": plan, "plan_sha256": sha(plan_path),
                "source_sha256": {str(path.relative_to(root)): sha(path) for path in sources},
                "input_sha256": {str(path.relative_to(root)): sha(path) for path in
                                 (root / "data/raw/linkedin.zip", root / "data/raw/yahoo.tar.gz", root / "config.json")},
                "output_sha256": {"results/tables/" + filename: sha(root / "results/tables" / filename) for filename in outputs},
                "original_table_sha256": original_tables, "original_tables_unchanged": True,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "fits": len(validation), "models": len(results), "threshold_folds": len(thresholds),
                "long_threshold_cells": len(long), "cross_dataset_budgets": len(cross),
                "metrics_omitted": plan["omitted_metrics"],
                "scope": "Target construction excludes held histograms; each fold target differs. No identified candidates, independent users, prospective data, human structure, population CI, or causal inference."}
    (root / "results/statistics/journal_extensions_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(results[["model", "held_log_loss", "constant_log_loss", "length15_observed_to_predicted_ratio", "length16_observed_to_predicted_ratio"]].to_string(index=False))
    print(json.dumps({"new_tables": len(outputs), "fits": len(validation), "long_cells": len(long),
                      "original_tables_unchanged": True}))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
