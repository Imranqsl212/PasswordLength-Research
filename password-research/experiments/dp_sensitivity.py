"""Aggregate DP diagnostics and conditional L1-radius illustrations.

No plaintext, noise reconstruction, iid-noise model, or confidence interval.
Mechanism: Blocki et al. NDSS 2016, DOI 10.14722/ndss.2016.23328.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis import read_sources, merge_histograms
from src.core import coverage, mass, size

KAPPA = math.pi * math.sqrt(2 / 3)
CONFIG = {
    "epsilon_per_list": 0.25,
    "reported_privacy_delta_upper": 2.0 ** -100,
    "utility_tail_probability_beta": 0.05,
    "assumed_true_mass_multipliers": [1, 2],
    "fixed_l1_radii": [1000, 10000, 100000, 1000000],
    "primary_lengths": list(range(6, 17)),
    "coverage_budget": 10000,
    "mechanism_status": "published restricted partition exponential mechanism; not reproduced",
    "radius_status": "conditional theoretical mass-proxy illustration; not actual noise or CI",
    "restriction_status": "true input partition is retained; restricted normalizer is at least one; no eta estimate used",
}


def spectrum_l1(first, second):
    """Exact zero-padded sorted-vector L1 without expanding multiplicities."""
    a, b = iter(first), iter(second)
    fa, ma = next(a, (0, 0))
    fb, mb = next(b, (0, 0))
    total = 0
    while ma or mb:
        if not ma:
            total += fb * mb + sum(f * m for f, m in b)
            break
        if not mb:
            total += fa * ma + sum(f * m for f, m in a)
            break
        take = min(ma, mb)
        total += take * abs(fa - fb)
        ma -= take
        mb -= take
        if not ma:
            fa, ma = next(a, (0, 0))
        if not mb:
            fb, mb = next(b, (0, 0))
    return total


def fraction_envelope(count, denominator, radius):
    """Conservative indicator-count fraction bounds under zero-padded L1<=E."""
    if not 0 <= count <= denominator or radius < 0 or denominator <= 0:
        raise ValueError("invalid count, denominator, or radius")
    if radius >= denominator:
        return 0.0, 1.0
    return (max(0.0, (count - radius) / (denominator + radius)),
            min(1.0, (count + radius) / (denominator - radius)))


def required_radius(count, denominator, target_fraction):
    """Necessary L1 radius to lower an indicator proportion to target.

    This bound permits the positive-bin denominator to change. It is not a
    sufficient radius or a proposed inverse mechanism.
    """
    if not 0 <= count <= denominator or denominator <= 0 or not 0 <= target_fraction <= 1:
        raise ValueError("invalid target")
    return max(0, math.ceil((count - target_fraction * denominator) / (1 + target_fraction)))


def log_tail_bound(assumed_mass, threshold, epsilon=0.25):
    """New finite upper bound from Lemma 4 and a partition-count bound.

    True input mass is assumed fixed and equal to, or at most, assumed_mass.
    For k=pi*sqrt(2/3), a=eps/2-k/(2sqrt(M))>0:
      P(L1>=T) <= 2exp(k sqrt(M)-a ceil(T))/(1-exp(-a)).
    This also bounds the published restricted mechanism directly: its support
    retains the input f (weight one), so its normalizer is at least one.
    Return natural log capped at zero; no finite bound if a<=0.
    """
    if assumed_mass <= 0 or threshold < 0 or epsilon <= 0:
        raise ValueError("invalid tail parameters")
    a = epsilon / 2 - KAPPA / (2 * math.sqrt(assumed_mass))
    if a <= 0:
        return None
    value = (math.log(2) + KAPPA * math.sqrt(assumed_mass)
             - a * math.ceil(threshold) - math.log(-math.expm1(-a)))
    return min(0.0, value)


def illustrative_radius(assumed_mass, epsilon=0.25, beta=0.05):
    """Integer E with derived P(L1>=E)<=beta at an ASSUMED fixed mass.

    Using E itself in envelopes is one unit conservative. A released-mass
    plug-in is an illustration, not a coverage guarantee conditional on data.
    """
    if assumed_mass <= 0 or epsilon <= 0 or not 0 < beta < 1:
        raise ValueError("invalid radius parameters")
    a = epsilon / 2 - KAPPA / (2 * math.sqrt(assumed_mass))
    if a <= 0:
        return None
    numerator = (math.log(2) + KAPPA * math.sqrt(assumed_mass)
                 - math.log(-math.expm1(-a)) - math.log(beta))
    return max(0, math.ceil(numerator / a))


def metric_counts(hist):
    return {"frequency1": sum(m for f, m in hist if f == 1),
            "frequency2": sum(m for f, m in hist if f == 2),
            "positive_even": sum(m for f, m in hist if f > 0 and f % 2 == 0)}


def save_table(root, name, rows):
    path = root / "results/tables" / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run(root=ROOT):
    if root.resolve() != ROOT.resolve():
        raise ValueError("source reader is bound to this project root")
    lengths, full, yahoo = read_sources()
    pool = merge_histograms(lengths.values())
    groups = {str(length): hist for length, hist in lengths.items()}
    groups.update({"LinkedIn_full": full, "LinkedIn_pooled_lengths": pool, "Yahoo_all": yahoo})
    paths = []

    parity = []
    bands = [(1, None), (1, 2), (3, 10), (11, 100), (101, 1000), (1001, None),
             (3, None), (5, None), (100, None)]
    for label, hist in groups.items():
        for lower, upper in bands:
            subset = [(f, m) for f, m in hist if f >= lower and (upper is None or f <= upper)]
            d, n = size(subset), mass(subset)
            if not d:
                continue
            even_d = sum(m for f, m in subset if f % 2 == 0)
            even_n = sum(f * m for f, m in subset if f % 2 == 0)
            parity.append({"group": label, "minimum_frequency": lower, "maximum_frequency": upper,
                           "anonymous_bins": d, "released_mass": n, "even_frequency_bins": even_d,
                           "even_bin_fraction": even_d / d, "even_account_fraction": even_n / n,
                           **metric_counts(subset)})
    paths.append(save_table(root, "dp_parity.csv", parity))

    pairs = []
    for length in CONFIG["primary_lengths"]:
        h = dict(lengths[length])
        for odd in range(1, 40, 2):
            first, second = h.get(odd, 0), h.get(odd + 1, 0)
            pairs.append({"length": length, "odd_frequency": odd, "even_frequency": odd + 1,
                          "odd_multiplicity": first, "even_multiplicity": second,
                          "even_to_odd_count_ratio": second / first if first else None,
                          "pair_even_fraction": second / (first + second) if first + second else None})
    paths.append(save_table(root, "dp_frequency_pairs.csv", pairs))

    paths.append(save_table(root, "dp_consistency.csv", [{
        "comparison": "LinkedIn_full_vs_sum_of_all_length_lists",
        "full_mass": mass(full), "pooled_mass": mass(pool), "mass_difference": mass(pool) - mass(full),
        "full_bins": size(full), "pooled_bins": size(pool), "bin_difference": size(pool) - size(full),
        "sorted_spectrum_l1": spectrum_l1(full, pool),
        "interpretation": "released-spectrum consistency; not actual noise size or source linkage",
    }]))

    sensitivities = []
    for length in CONFIG["primary_lengths"]:
        hist = lengths[length]
        d, n = size(hist), mass(hist)
        counts = metric_counts(hist)
        selected_mass = coverage(hist, CONFIG["coverage_budget"]) * n
        settings = [("fixed_radius", radius, None) for radius in CONFIG["fixed_l1_radii"]]
        for multiplier in CONFIG["assumed_true_mass_multipliers"]:
            assumed = multiplier * n
            radius = illustrative_radius(assumed, beta=CONFIG["utility_tail_probability_beta"])
            settings.append(("theoretical_mass_proxy", radius, assumed))
        for status, radius, assumed in settings:
            row = {"length": length, "status": status, "assumed_true_mass": assumed,
                   "utility_beta": CONFIG["utility_tail_probability_beta"] if assumed else None,
                   "l1_radius": radius, "released_mass": n, "anonymous_bins": d,
                   "radius_fraction_of_released_mass": radius / n if radius is not None else None}
            if radius is not None:
                for metric, count in counts.items():
                    lo, hi = fraction_envelope(count, d, radius)
                    row.update({metric + "_observed": count / d, metric + "_lower": lo, metric + "_upper": hi})
                lo, hi = fraction_envelope(selected_mass, n, radius)
                row.update({"conditional_top10000_account_observed": selected_mass / n,
                            "conditional_top10000_account_lower": lo, "conditional_top10000_account_upper": hi})
            sensitivities.append(row)
    paths.append(save_table(root, "dp_radius_sensitivity.csv", sensitivities))

    h15 = lengths[15]
    d15, n15 = size(h15), mass(h15)
    c15 = metric_counts(h15)
    comparisons = []
    for metric in ("frequency2", "positive_even"):
        targets = [("length14_released_fraction", metric_counts(lengths[14])[metric] / size(lengths[14])),
                   ("length16_released_fraction", metric_counts(lengths[16])[metric] / size(lengths[16])),
                   ("illustrative_half", 0.5)]
        for label, target in targets:
            required = required_radius(c15[metric], d15, target)
            log_bound = log_tail_bound(2 * n15, required)
            comparisons.append({"length": 15, "metric": metric, "observed_fraction": c15[metric] / d15,
                                "target": label, "target_fraction": target,
                                "necessary_l1_radius_allowing_bin_count_change": required,
                                "necessary_l1_radius_fixed_bin_count": max(0, math.ceil(c15[metric] - target * d15)),
                                "required_radius_fraction_of_released_mass": required / n15,
                                "assumed_true_mass_for_tail": 2 * n15,
                                "conditional_log10_tail_upper_at_required_radius": log_bound / math.log(10) if log_bound is not None else None,
                                "status": "necessary distance; conditional mechanism bound, not attribution or a p-value"})
    paths.append(save_table(root, "dp_noise_lower_bounds.csv", comparisons))

    robustness = pd.read_csv(root / "results/tables/robustness.csv")
    original = robustness[robustness.condition == "all"].iloc[0]
    cutoffs = []
    for cutoff in ("all", "frequency_at_least_2", "frequency_at_least_5"):
        row = robustness[robustness.condition == cutoff].iloc[0]
        cutoffs.append({"condition": cutoff, "retained_bins": row.anonymous_distinct_bins,
                        "removed_bin_fraction": 1 - row.anonymous_distinct_bins / original.anonymous_distinct_bins,
                        "retained_account_mass": row.released_account_mass,
                        "removed_account_mass_fraction": 1 - row.released_account_mass / original.released_account_mass,
                        "rho_length_frequency_distinct": row.rho_length_frequency_distinct,
                        "rho_length_frequency_account": row.rho_length_frequency_account,
                        "status": "changes estimand; not estimated additive noise or causal DP effect"})
    paths.append(save_table(root, "dp_cutoff_estimands.csv", cutoffs))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_AGGREGATE_DIAGNOSTICS_AND_ILLUSTRATIVE_RADIUS_ANALYSIS",
        "configuration": CONFIG,
        "configuration_sha256": hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
        "input_sha256": {str(p.relative_to(root)): sha(p) for p in
                         (root / "data/raw/linkedin.zip", root / "data/raw/yahoo.tar.gz", root / "results/tables/robustness.csv")},
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in
                          (Path(__file__), root / "src/core.py", root / "src/analysis.py")},
        "output_sha256": {str(p.relative_to(root)): sha(p) for p in paths},
        "primary_sources": ["https://eprint.iacr.org/2016/153.pdf", "https://arxiv.org/pdf/2002.01513"],
        "assumptions": [
            "Integer sorted spectra with zero padding; distance is half L1.",
            "Unknown true mass is illustrated at released mass and twice released mass.",
            "No probability guarantee after plugging a data-dependent released mass into the bound.",
            "Restricted normalizer is at least one because the true input f is retained; no removed-support probability eta is inferred from privacy delta.",
            "Radius envelopes condition on chosen L1 budget; neither sharp intervals nor noise confidence intervals.",
            "Utility beta=.05 is separate from privacy delta and is not a simultaneous per-length guarantee.",
        ],
        "not_performed": ["plaintext reconstruction", "noise reconstruction", "iid Laplace simulation",
                          "published mechanism reimplementation", "actual-noise estimation", "noise CI", "causal attribution"],
        "full_vs_pooled_l1": spectrum_l1(full, pool),
    }
    (root / "results/statistics/dp_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"tables": len(paths), "full_vs_pooled_l1": metadata["full_vs_pooled_l1"],
                      "length15_radius_at_released_mass": illustrative_radius(n15),
                      "length15_radius_at_twice_released_mass": illustrative_radius(2 * n15)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
