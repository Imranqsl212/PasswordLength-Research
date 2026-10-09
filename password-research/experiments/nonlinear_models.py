#!/usr/bin/env python3
"""Exploratory length-only calibration diagnostics on released aggregate bins.

No password strings, guessing candidates, or synthetic regeneration are used.
Targets match the original pooled top-K diagnostic, including its full-pool
cutoff limitation. Basis definitions below are frozen before this comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer


REGULARIZATION_C = 1e6
OPTIMIZER_TOLERANCE = 1e-10
MAX_ITERATIONS = 10_000
PROBABILITY_EPSILON = 1e-15
LEGACY_PREDICTION_RTOL = 1e-8
LEGACY_PREDICTION_ATOL = 1e-12


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    degree: int = 1
    n_knots: int = 0

    def definition(self) -> dict:
        return {"name": self.name, "family": self.family, "degree": self.degree,
                "n_knots": self.n_knots,
                "knots": "uniform across training min/max" if self.n_knots else None,
                "extrapolation": "linear" if self.n_knots else "polynomial" if self.family == "polynomial" else None,
                "include_bias": False,
                "standardization": "training-only distinct-bin-weighted mean and SD of source-defined length",
                "feature_rescaling": "none after forming the fixed basis"}


MODEL_SPECS = (
    ModelSpec("intercept_only", "intercept", degree=0),
    ModelSpec("linear_logistic", "polynomial", degree=1),
    ModelSpec("quadratic_logistic", "polynomial", degree=2),
    ModelSpec("cubic_logistic", "polynomial", degree=3),
    ModelSpec("bspline_cubic_4knots", "bspline", degree=3, n_knots=4),
    ModelSpec("bspline_cubic_5knots", "bspline", degree=3, n_knots=5),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_groups(histogram: pd.DataFrame, primary_lengths: list[int], budget: int) -> tuple[pd.DataFrame, dict]:
    """Collapse anonymous frequencies to exact sufficient group statistics."""
    required = {"length", "frequency", "multiplicity"}
    if not required <= set(histogram):
        raise ValueError("Expected released length/frequency/multiplicity columns")
    values = histogram[list(required)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values != np.floor(values)):
        raise ValueError("Released histogram values must be finite positive integers")
    if not primary_lengths or len(set(primary_lengths)) != len(primary_lengths):
        raise ValueError("Primary lengths must be a nonempty unique list")
    pool = Counter()
    for row in histogram.itertuples():
        pool[int(row.frequency)] += int(row.multiplicity)
    total_bins = sum(pool.values())
    if not isinstance(budget, int) or not 1 <= budget <= total_bins:
        raise ValueError("Budget must be within the released pooled bin count")
    selected_fraction = {}
    first_rank = 1
    cutoff = None
    for frequency, multiplicity in sorted(pool.items(), reverse=True):
        selected = max(0, min(multiplicity, budget - first_rank + 1))
        selected_fraction[frequency] = selected / multiplicity
        if selected:
            cutoff = frequency
        first_rank += multiplicity
    rows = []
    for length in sorted(primary_lengths):
        subset = histogram[histogram.length == length]
        if subset.empty:
            raise ValueError(f"Missing required released length group {length}")
        count = int(subset.multiplicity.sum())
        selected_mass = math.fsum(int(row.multiplicity) * selected_fraction[int(row.frequency)]
                                  for row in subset.itertuples())
        rows.append({"length": length, "anonymous_bins": count,
                     "fractional_early_bins": selected_mass,
                     "observed_distinct_early_fraction": selected_mass / count})
    return pd.DataFrame(rows), {
        "pooled_anonymous_bins": total_bins, "pooled_length_groups": int(histogram.length.nunique()),
        "pooled_topk_cutoff_frequency": cutoff, "target_budget": budget,
        "target_cutoff_scope": "full released pool, including each held group",
        "tie_rule": "fractional membership shared uniformly within a pooled equal-frequency boundary",
        "compression": "two weighted binary pseudo-observations per length; exact sufficient statistics for length-only Bernoulli objective",
    }


def binary_design(groups: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, weight = [], [], []
    for row in groups.itertuples():
        for label, mass in ((1, row.fractional_early_bins),
                            (0, row.anonymous_bins - row.fractional_early_bins)):
            if mass > 0:
                x.append(float(row.length)); y.append(label); weight.append(float(mass))
    return np.asarray(x), np.asarray(y), np.asarray(weight)


class LengthBasis:
    """A transformer whose location, scale, and knots see training inputs only."""
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.spline = None

    def fit(self, x: np.ndarray, weights: np.ndarray) -> "LengthBasis":
        if len(x) == 0 or len(x) != len(weights) or np.any(weights <= 0):
            raise ValueError("A basis needs nonempty positive-weight training inputs")
        self.mean = float(np.average(x, weights=weights))
        self.sd = math.sqrt(float(np.average((x - self.mean) ** 2, weights=weights))) or 1.0
        self.min_length, self.max_length = float(min(x)), float(max(x))
        z = ((x - self.mean) / self.sd).reshape(-1, 1)
        if self.spec.family == "bspline":
            self.spline = SplineTransformer(
                n_knots=self.spec.n_knots, degree=self.spec.degree, knots="uniform",
                extrapolation="linear", include_bias=False,
            ).fit(z)
        elif self.spec.family != "polynomial":
            raise ValueError("Expected polynomial or B-spline basis")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self.mean) / self.sd
        if self.spline is not None:
            return self.spline.transform(z.reshape(-1, 1))
        return np.column_stack([z ** degree for degree in range(1, self.spec.degree + 1)])

    def details(self) -> dict:
        result = {"training_weighted_mean": self.mean, "training_weighted_sd": self.sd,
                  "training_min_length": self.min_length, "training_max_length": self.max_length}
        if self.spline is not None:
            knots = self.spline.bsplines_[0].t
            result["spline_knot_vector_source_units"] = (knots * self.sd + self.mean).tolist()
        return result


@dataclass
class FittedModel:
    spec: ModelSpec
    basis: LengthBasis | None
    classifier: LogisticRegression | None
    constant: float | None
    convergence_warning: str
    iterations: int

    def predict(self, lengths: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            return np.full(len(lengths), self.constant, dtype=float)
        return self.classifier.predict_proba(self.basis.transform(lengths))[:, 1]

    def parameters(self) -> dict:
        if self.classifier is None:
            return {"constant_probability": self.constant,
                    "intercept": math.log(self.constant / (1 - self.constant))}
        return {"intercept": float(self.classifier.intercept_[0]),
                "basis_coefficients": self.classifier.coef_[0].tolist(), **self.basis.details()}


def fit_model(spec: ModelSpec, groups: pd.DataFrame) -> FittedModel:
    x, y, weights = binary_design(groups)
    if len(set(y)) != 2:
        raise ValueError("Both positive and negative target mass are required")
    if spec.family == "intercept":
        constant = float(np.average(y, weights=weights))
        return FittedModel(spec, None, None, constant, "", 0)
    basis = LengthBasis(spec).fit(x, weights)
    classifier = LogisticRegression(C=REGULARIZATION_C, max_iter=MAX_ITERATIONS,
                                    tol=OPTIMIZER_TOLERANCE, solver="lbfgs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(basis.transform(x), y, sample_weight=weights / weights.sum())
    warning_text = "; ".join(str(item.message) for item in caught
                              if issubclass(item.category, ConvergenceWarning))
    return FittedModel(spec, basis, classifier, None, warning_text, int(classifier.n_iter_[0]))


def scoring(observed: float, predicted: float) -> dict[str, float]:
    if not 0 <= observed <= 1 or not 0 <= predicted <= 1:
        raise ValueError("Scores require probabilities in [0,1]")
    p = float(np.clip(predicted, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
    return {
        "log_loss": -(observed * math.log(p) + (1 - observed) * math.log1p(-p)),
        "brier_loss": observed * (1 - p) ** 2 + (1 - observed) * p ** 2,
        "abs_calibration_error": abs(observed - p),
        "squared_calibration_error": (observed - p) ** 2,
        "observed_to_predicted_ratio": observed / p,
    }


def validation_row(model: FittedModel, row, train: pd.DataFrame, scope: str) -> dict:
    p = float(model.predict(np.asarray([row.length], dtype=float))[0])
    q = float(row.observed_distinct_early_fraction)
    return {"model": model.spec.name, "fit_scope": scope,
            "fit_id": "full_data" if scope == "full_data" else f"held_length_{int(row.length)}",
            "held_length": int(row.length),
            "observed_distinct_early_fraction": q, "predicted_fraction": p,
            **scoring(q, p), "test_anonymous_bins": int(row.anonymous_bins),
            "training_anonymous_bins": int(train.anonymous_bins.sum()),
            "constant_training_fraction": float(train.fractional_early_bins.sum() / train.anonymous_bins.sum()),
            "training_min_length": int(train.length.min()), "training_max_length": int(train.length.max()),
            "feature_count": 0 if model.classifier is None else int(model.classifier.n_features_in_),
            "iterations": model.iterations, "convergence_warning": model.convergence_warning,
            "fitted_basis_details": json.dumps(model.parameters(), sort_keys=True)}


def evaluate(groups: pd.DataFrame, specs=MODEL_SPECS) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation, summary = [], []
    for spec in specs:
        full = fit_model(spec, groups)
        for row in groups.itertuples():
            validation.append(validation_row(full, row, groups, "full_data"))
        for row in groups.itertuples():
            training = groups[groups.length != row.length]
            fold = fit_model(spec, training)
            validation.append(validation_row(fold, row, training, "held_length"))
        current = pd.DataFrame([row for row in validation if row["model"] == spec.name])
        result = {"model": spec.name, "basis_definition": json.dumps(spec.definition(), sort_keys=True),
                  "anonymous_bins": int(groups.anonymous_bins.sum()), "length_groups": len(groups),
                  "feature_count_full": 0 if full.classifier is None else int(full.classifier.n_features_in_),
                  "full_parameters": json.dumps(full.parameters(), sort_keys=True),
                  "C": REGULARIZATION_C if spec.family != "intercept" else None,
                  "target_weighting": "distinct anonymous bins",
                  "structural_model_status": "DATA UNAVAILABLE",
                  "validation": "leave-length-out descriptive diagnostic; full-pool target labels include held group",
                  "convergence_warning_count": int(sum(bool(value) for value in current.drop_duplicates("fit_id").convergence_warning)),
                  "max_iterations_used": int(current.iterations.max())}
        for scope in ("full_data", "held_length"):
            selected = current[current.fit_scope == scope]
            for metric in ("log_loss", "brier_loss", "abs_calibration_error", "squared_calibration_error"):
                result[f"{scope}_{metric}"] = float(np.average(selected[metric], weights=selected.test_anonymous_bins))
            result[f"{scope}_equal_group_log_loss"] = float(selected.log_loss.mean())
            for length in (15, 16):
                selected_length = selected[selected.held_length == length]
                result[f"{scope}_length{length}_observed_to_predicted_ratio"] = (
                    float(selected_length.observed_to_predicted_ratio.iloc[0]) if len(selected_length) else None)
        summary.append(result)
    result = pd.DataFrame(summary)
    if {"intercept_only", "linear_logistic"} <= set(result.model):
        indexed = result.set_index("model")
        for baseline in ("intercept_only", "linear_logistic"):
            baseline_loss = float(indexed.loc[baseline, "held_length_log_loss"])
            result[f"held_length_loss_ratio_to_{baseline}"] = result.held_length_log_loss / baseline_loss
    return result, pd.DataFrame(validation)


def compare_legacy(validation: pd.DataFrame, legacy: pd.DataFrame) -> dict:
    new = validation[(validation.model == "linear_logistic") & (validation.fit_scope == "held_length")]
    merged = new.merge(legacy, on="held_length", suffixes=("_new", "_legacy"), validate="one_to_one")
    if len(merged) != len(legacy):
        raise AssertionError("Legacy and new linear diagnostics have different folds")
    pnew, pold = merged.predicted_fraction_new.to_numpy(), merged.predicted_fraction_legacy.to_numpy()
    qnew, qold = merged.observed_distinct_early_fraction_new.to_numpy(), merged.observed_distinct_early_fraction_legacy.to_numpy()
    if not np.allclose(qnew, qold, rtol=1e-12, atol=1e-15):
        raise AssertionError("Target differs from the existing released-data model")
    # Collapsing identical-input rows is objective-equivalent; solver accumulation can differ slightly.
    if not np.allclose(pnew, pold, rtol=LEGACY_PREDICTION_RTOL, atol=LEGACY_PREDICTION_ATOL):
        raise AssertionError("Linear baseline does not agree within the declared numerical tolerance")
    return {"status": "PASSED", "probability_rtol": LEGACY_PREDICTION_RTOL,
            "probability_atol": LEGACY_PREDICTION_ATOL,
            "max_absolute_prediction_difference": float(np.max(np.abs(pnew - pold))),
            "max_relative_prediction_difference": float(np.max(np.abs(pnew - pold) / pold)),
            "max_absolute_log_loss_difference": float(np.max(np.abs(merged.log_loss_new - merged.log_loss_legacy))),
            "reason_for_possible_difference": "exact sufficient-statistic row collapse and floating-point optimizer accumulation"}


def report(root: Path, summary: pd.DataFrame, validation: pd.DataFrame, agreement: dict) -> None:
    lines = ["# Exploratory nonlinear length-only diagnostics", "",
             "This comparison was requested after inspecting the original linear model's miscalibration at source-defined lengths 15 and 16. It is an exploratory post-result revision, not a preregistered confirmatory test. The six model/basis definitions were fixed in code before executing this new comparison; no basis, degree, knot count or regularization was selected by held-fold performance.", "",
             "## Target and weighting", "",
             "Inputs are the existing released length–frequency–multiplicity histogram. The outcome matches the original model: fractional membership in the pooled top 10,000 anonymous distinct bins. A boundary frequency tie is fractionally shared across its entire pooled multiplicity. Each distinct anonymous bin has one weight, irrespective of its released frequency; these are not account-weighted recovery percentages. Frequencies are pooled over all released length groups, while model fitting covers source-defined groups 6–16.", "",
             "The cutoff and outcome labels use the complete pool, including the held-out group. Leave-one-length-group-out results are descriptive calibration diagnostics, not prospective or independent validation. Individual strings, generation mechanisms, encoding and normalization are unobserved. Source groups are not a verified human-generated cohort. A human length-plus-structure model remains **DATA UNAVAILABLE**.", "",
             "## Fixed model definitions", "",
             "The intercept-only model estimates the training weighted early-bin fraction directly. Logistic models use training-only distinct-bin-weighted length mean/SD. The linear, quadratic and cubic bases are z, (z,z²) and (z,z²,z³). Cubic B-splines use four or five uniform knots spanning the training min/max, omit the redundant bias basis, and explicitly extrapolate linearly beyond the training span. The basis transformer is fitted anew inside every training fold. Knot positions, means and SDs for every fit are saved with validation rows.", "",
             f"All logistic models use scikit-learn LBFGS, C={REGULARIZATION_C:g}, tolerance {OPTIMIZER_TOLERANCE:g}, maximum {MAX_ITERATIONS:,} iterations, and training weights normalized to sum one, as in the original linear diagnostic. Derived basis columns receive no extra rescaling. A common C is not parameterization-invariant; basis scale and collinearity affect the penalty, so this is a fixed diagnostic comparison, not a claim that complexity was optimally tuned. Two weighted binary pseudo-observations per length exactly preserve sufficient statistics of the original frequency-row objective; no synthetic inputs were regenerated.", "",
             "Library semantics are documented by the official [SplineTransformer API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) and [LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html); actual installed versions are recorded in run metadata.", "",
             "## Observed results", "",
             "Losses are weighted by held-group distinct-bin counts, then pooled over all eleven folds. Full-data metrics are in-sample descriptions. Bernoulli log loss and Brier loss use the fractional positive mass plus negative mass; Brier loss is not merely the squared difference between group prevalence and prediction. Calibration absolute error is a separate group-prevalence metric. Small aggregate improvements can coexist with poor endpoint calibration because long groups have relatively small distinct-bin weights.", "",
             "| Model | Full-data log loss | Held-length log loss | Held Brier loss | Held weighted absolute calibration error | L15 observed/predicted | L16 observed/predicted |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.model} | {row.full_data_log_loss:.9f} | {row.held_length_log_loss:.9f} | {row.held_length_brier_loss:.9f} | {row.held_length_abs_calibration_error:.9f} | {row.held_length_length15_observed_to_predicted_ratio:.3f} | {row.held_length_length16_observed_to_predicted_ratio:.3f} |")
    linear = float(summary.set_index("model").loc["linear_logistic", "held_length_log_loss"])
    lines += ["", "Values above 1 in observed/predicted ratios indicate underprediction; values below 1 indicate overprediction. Neither the whole-pool label construction nor differential-privacy release noise supplies an independent person-level sampling model, so no population p-values or confidence intervals are calculated.", "",
              "| Nonlinear model | Held log-loss change relative to linear (%) | Endpoint interpretation |", "|---|---:|---|"]
    for row in summary[~summary.model.isin(["intercept_only", "linear_logistic"])].itertuples():
        change = 100 * (row.held_length_log_loss / linear - 1)
        ratio15 = row.held_length_length15_observed_to_predicted_ratio
        ratio16 = row.held_length_length16_observed_to_predicted_ratio
        interpretation = f"L15 ratio {ratio15:.3f}; L16 ratio {ratio16:.3f}; retained as descriptive checks"
        lines.append(f"| {row.model} | {change:+.4f} | {interpretation} |")
    lines += ["", "Every fixed model is reported, including failures to improve. The four/five-knot comparison is a sensitivity check, not a basis selection rule. A held group at 16 lies outside training support (maximum 15), making its result explicit extrapolation; group 15 is interior when group 16 remains in training. Better full-data fitting does not establish better held-group prediction, realistic password guessability, causality, or generalization to current services.", "",
              "## Baseline agreement and reproducibility", "",
              f"The collapsed linear diagnostic agrees with the existing model within declared rtol={agreement['probability_rtol']:g} and atol={agreement['probability_atol']:g}. Maximum absolute probability difference was {agreement['max_absolute_prediction_difference']:.12g}; maximum relative probability difference was {agreement['max_relative_prediction_difference']:.12g}. Maximum per-fold log-loss difference was {agreement['max_absolute_log_loss_difference']:.12g}. Tiny numerical differences arise from equivalent row collapse and optimizer accumulation; the fractional targets match to rtol 1e-12.", "",
              "Run from the project directory:", "", "```sh", "../.venv/bin/python experiments/nonlinear_models.py --root .", "../.venv/bin/python -m unittest discover -s tests -p test_nonlinear_models.py", "```", "",
              "Outputs: `results/tables/nonlinear_model_results.csv` (all model summaries), `results/tables/nonlinear_model_validation.csv` (full-data and held-length calibration rows), and `results/statistics/nonlinear_metadata.json` (frozen definitions, source/config/input/output hashes, versions, convergence and legacy agreement). Filter validation rows with `fit_scope == 'held_length'` for held-fold figures; do not mix them with `full_data` rows. Existing source tables and synthetic controls are read only.", ""]
    path = root / "docs/nonlinear-models.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    histogram_path = root / "data/derived/length_frequency_histogram.csv"
    legacy_path = root / "results/tables/model_validation.csv"
    groups, target_metadata = prepare_groups(pd.read_csv(histogram_path), config["primary_lengths"], config["primary_budget"])
    summary, validation = evaluate(groups)
    agreement = compare_legacy(validation, pd.read_csv(legacy_path))
    summary.insert(1, "budget", config["primary_budget"])
    results_path = root / "results/tables/nonlinear_model_results.csv"
    validation_path = root / "results/tables/nonlinear_model_validation.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_path, index=False)
    validation.to_csv(validation_path, index=False)
    report(root, summary, validation, agreement)
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_status": "exploratory post-result revision; fixed basis choices before this comparison; no held-fold tuning",
        "source_sha256": sha(Path(__file__).resolve()), "config_sha256": sha(config_path),
        "input_sha256": {str(path.relative_to(root)): sha(path) for path in (histogram_path, legacy_path)},
        "output_sha256": {str(path.relative_to(root)): sha(path) for path in (results_path, validation_path, root / "docs/nonlinear-models.md")},
        "frozen_model_definitions": [spec.definition() for spec in MODEL_SPECS],
        "optimizer": {"C": REGULARIZATION_C, "solver": "lbfgs", "tol": OPTIMIZER_TOLERANCE,
                      "max_iter": MAX_ITERATIONS, "weight_normalization": "training sum one"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "target": target_metadata, "primary_lengths": config["primary_lengths"],
        "primary_anonymous_bins": int(groups.anonymous_bins.sum()),
        "linear_legacy_agreement": agreement, "structural_model_status": "DATA UNAVAILABLE",
        "fit_rows": len(validation), "models": len(MODEL_SPECS),
        "fits": len(validation.drop_duplicates(["model", "fit_id"])),
        "fits_with_convergence_warning": int(sum(bool(value) for value in validation.drop_duplicates(["model", "fit_id"]).convergence_warning)),
        "limitations": ["Target cutoff includes held group; no prospective validation", "No verified human-generation labels or structural features", "Source-defined length encoding and normalization unknown", "Released DP spectra are descriptive", "Held L16 is extrapolation", "C is basis-scale dependent", "No hyperparameter search or held-fold model selection"],
    }
    metadata_path = root / "results/statistics/nonlinear_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary[["model", "full_data_log_loss", "held_length_log_loss", "held_length_length15_observed_to_predicted_ratio", "held_length_length16_observed_to_predicted_ratio"]].to_string(index=False))
    print(json.dumps({"metadata": str(metadata_path), "legacy_agreement": agreement}))
    return summary, validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.root.resolve())
