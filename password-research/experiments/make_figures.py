#!/usr/bin/env python3
"""Draw publication figures from saved aggregate tables, without password inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Keep plotting/font caches writable on sandboxed hosts; preserve explicit user paths.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "password-research-matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "password-research-font-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator
import numpy as np
import pandas as pd


COLORS = ["#176B87", "#D06A29", "#238673", "#845CA2", "#BC4752", "#59734A"]
INPUT_FILES = (
    "theoretical_baseline.csv", "length_distribution.csv", "coverage_curves.csv",
    "length_bins.csv", "model_validation.csv", "transformations.csv",
    "synthetic_summary.csv", "passphrase_summary.csv", "passphrase_conditional.csv",
    "hashing_cost.csv", "nonlinear_model_validation.csv",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "figure.titlesize": 16, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    })


def format_log_axis(ax, which="y") -> None:
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10, numticks=7))
    axis.set_major_formatter(LogFormatterMathtext(base=10))


def new_figure(title: str, panels: int = 1, width: float = 10.4):
    fig, axes = plt.subplots(1, panels, figsize=(width, 5.2), squeeze=False)
    fig.suptitle(title, x=.065, ha="left", y=.98, weight="bold")
    fig.subplots_adjust(left=.075, right=.97, top=.82, bottom=.20, wspace=.28)
    return fig, axes.ravel()


def save(fig, out: Path, stem: str, footer: str) -> list[str]:
    fig.text(.065, .055, footer, ha="left", va="bottom", fontsize=8, color="#45515A")
    files = []
    for extension in ("png", "pdf", "svg"):
        path = out / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.12)
        files.append(path.name)
    plt.close(fig)
    return files


def run(root: Path) -> None:
    setup_style()
    table_dir = root / "results/tables"
    out = root / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    tables = {name: pd.read_csv(table_dir / name) for name in INPUT_FILES}
    lengths = tables["length_distribution.csv"].copy()
    primary = lengths[lengths.length.between(6, 16)].sort_values("length")
    if primary.length.tolist() != list(range(6, 17)):
        raise ValueError("All source lengths 6–16 are required; do not omit anomalous groups")
    primary_mass = int(primary.released_account_mass.sum())
    primary_bins = int(primary.anonymous_distinct_bins.sum())
    theory = tables["theoretical_baseline.csv"]
    curves = tables["coverage_curves.csv"]
    hashing = tables["hashing_cost.csv"]
    captions = []
    files = []

    # Figure 1: an explicitly mathematical baseline, without observed passwords.
    fig, (ax,) = new_figure("Figure 1. Uniform search space grows with character length")
    alphabet_names = {"lowercase26": "Lowercase (N=26)", "letters52": "Letters (N=52)",
                      "alphanumeric62": "Alphanumeric (N=62)", "printable94": "Printable non-space ASCII (N=94)"}
    for name, label in alphabet_names.items():
        data = theory[theory.alphabet == name].sort_values("length")
        y = [math.log10(int(value)) for value in data.search_space]
        ax.plot(data.length, y, marker="o", markersize=3.5, label=label)
    ax.set(xlabel="Character length L", ylabel="log₁₀ of the number of possible strings Nᴸ")
    ax.set_xticks([6, 8, 10, 12, 16, 20, 24, 32])
    ax.legend(loc="upper left", frameon=False)
    files += save(fig, out, "fig01_theoretical_search_space",
                  "Theoretical distribution: independent uniform draws. No human distribution or empirical sample size is implied.")
    captions.append((1, "fig01_theoretical_search_space", "Theoretical",
        "Mathematical search spaces N^L for four explicitly defined alphabets and all configured lengths. The vertical axis is log10 search-space size; the spacing therefore represents orders of magnitude. Printable ASCII uses 94 visible characters, excluding space. This calculation assumes independent uniform choices and does not estimate entropy of observed human passwords. No empirical denominator applies."))

    # Figure 2: distinguish two weighting schemes, then the conditional oracle.
    fig, (ax, bx) = new_figure("Figure 2. Frequency and oracle coverage vary nonmonotonically", 2)
    ax.plot(primary.length, primary.account_weighted_median_frequency, "o-", label="Account-mass-weighted median")
    ax.plot(primary.length, primary.distinct_median_frequency, "s--", label="Distinct-bin median")
    ax.set(xlabel="Source-defined length group", ylabel="Median released frequency (occurrences)", yscale="log")
    ax.set_xticks(range(6, 17, 2)); ax.legend(frameon=False, loc="upper right")
    format_log_axis(ax)
    bx.plot(primary.length, primary.conditional_g50, "o-", color=COLORS[2])
    bx.set(xlabel="Source-defined length group", ylabel="Conditional G50 (anonymous oracle budget)", yscale="log")
    bx.set_xticks(range(6, 17, 2)); format_log_axis(bx)
    ax.set_title("Frequency medians depend on weighting", loc="left")
    bx.set_title("Length is known to the oracle", loc="left")
    files += save(fig, out, "fig02_frequency_and_g50",
                  f"Empirical released LinkedIn DP histograms, lengths 6–16: mass={primary_mass:,}; anonymous bins={primary_bins:,}.")
    captions.append((2, "fig02_frequency_and_g50", "Empirical aggregate-spectrum analysis",
        f"Left: released-frequency medians using either one vote per anonymous distinct bin or released account mass as the weight. Right: the minimum anonymous frequency-oracle budget sufficient to cover half of the mass within each known-length group (conditional G50). All eleven source groups 6–16 are included; no anomalous group is removed. Across these groups the released mass is {primary_mass:,} and the number of anonymous bins is {primary_bins:,}. Neither weighting estimates a representative population of all users. Values inherit differential-privacy release distortion, length-policy/source effects and unknown password identities."))

    # Figure 3: different denominators and oracle conditions occupy separate panels.
    fig, (ax, bx) = new_figure("Figure 3. Anonymous frequency-oracle cumulative coverage", 2, 11.2)
    selected = (6, 8, 10, 12, 14, 16)
    for i, length in enumerate(selected):
        data = curves[curves.group.astype(str) == str(length)].sort_values("guesses")
        ax.plot(data.guesses, 100 * data.account_coverage, label=f"L={length}", color=COLORS[i])
    comparator_labels = {"LinkedIn_full": "LinkedIn full release", "Yahoo_all": "Yahoo release",
                         "LinkedIn_pooled_lengths": "LinkedIn pooled length releases"}
    for i, (group, label) in enumerate(comparator_labels.items()):
        data = curves[curves.group == group].sort_values("guesses")
        bx.plot(data.guesses, 100 * data.account_coverage, label=label, color=COLORS[i],
                linestyle="--" if group == "LinkedIn_pooled_lengths" else "-")
    for axis in (ax, bx):
        axis.set(xscale="log", xlabel="Anonymous oracle budget K", ylabel="Released account mass covered (%)", ylim=(0, 102))
        format_log_axis(axis, "x"); axis.legend(frameon=False, loc="upper left")
    ax.set_title("Conditional on the displayed length", loc="left")
    bx.set_title("Whole-release comparators", loc="left")
    denominator_text = "; ".join(f"{label}: {int(curves[curves.group == group].denominator_mass.iloc[0]):,}"
                                 for group, label in comparator_labels.items())
    files += save(fig, out, "fig03_oracle_coverage",
                  "Each curve uses its own released account-mass denominator. Budgets stop at the available anonymous-bin count.")
    captions.append((3, "fig03_oracle_coverage", "Empirical aggregate-spectrum analysis",
        "Left: conditional anonymous frequency-oracle coverage for six evenly spaced displayed length groups; each denominator is its own group's released mass. Right: whole-release comparator curves, with denominators " + denominator_text + ". Every curve ends within its available number of anonymous bins. The pooled length releases and full LinkedIn release differ in mass because they are separate noisy releases; they are not independent corpora. Lines connect evaluated budgets for readability. Frequency spectra determine an idealized oracle upper bound, not an executable ordering of identified passwords or an actual account-recovery experiment."))

    # Figure 4: central comparison, carefully separate aggregate observation and theory.
    fig, (ax, bx) = new_figure("Figure 4. Equal length permits very different predictability", 2, 11.2)
    ax.plot(primary.length, primary.conditional_top10000_account_pct / 100,
            "o-", color=COLORS[0], label="Released account-password corpus:\nconditional oracle")
    for category, label, color in (("lowercase26", "Uniform lowercase: N=26", COLORS[2]),
                                   ("printable94", "Uniform printable: N=94", COLORS[1])):
        data = theory[(theory.alphabet == category) & theory.length.between(6, 16)].sort_values("length")
        ax.plot(data.length, data.coverage_at_10000, "s--", color=color, markersize=3, label=label)
    ax.set(xlabel="Source-defined length (nominal)", ylabel="Coverage probability at K=10,000 (log scale)", yscale="log", ylim=(1e-29, 1))
    ax.set_xticks(range(6, 17, 2)); format_log_axis(ax)
    ax.set_yticks([10. ** exponent for exponent in (0, -4, -8, -12, -16, -20, -24, -28)])
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Observed spectrum versus uniform mathematics", loc="left")
    synthetic = tables["synthetic_summary.csv"]
    for category, label, color, style in (
        ("lowercase26", "Independent lowercase characters", COLORS[2], "-"),
        ("printable94", "Independent printable characters", COLORS[1], "-"),
        ("synthetic_repeat2", "Repeated random 2-letter unit", COLORS[3], "--"),
        ("synthetic_fixed_suffix", "Random 4 letters + fixed suffix/padding", COLORS[4], "--"),
    ):
        data = synthetic[synthetic.category == category].groupby("length").entropy_bits.first().sort_index()
        bx.plot(data.index, data.values, marker="o", markersize=3, color=color, linestyle=style, label=label)
    bx.set(xlabel="Character length", ylabel="Known generator entropy (bits)")
    bx.set_xticks([6, 8, 10, 12, 16, 20, 24, 32]); bx.legend(frameon=False, loc="upper left")
    bx.set_title("Structure controls: generator mathematics", loc="left")
    n_controls = int(synthetic[synthetic.category.isin(["lowercase26", "printable94", "synthetic_repeat2", "synthetic_fixed_suffix"])].n.sum())
    files += save(fig, out, "fig04_same_length_comparison",
                  f"Left: released corpus mass={primary_mass:,}; uniform probability is mathematical. Right: {n_controls:,} synthetic feature records; entropy is derived.")
    captions.append((4, "fig04_same_length_comparison", "Empirical and theoretical, explicitly distinguished",
        f"Left: observed released account-password corpus mass covered by an oracle knowing the source-defined length at budget K=10,000, alongside the mathematical probabilities min(K/N^L,1) for independently uniform N=26 and N=94 strings. Corpus coverage uses each released length group's account mass; combined mass is {primary_mass:,}. Generation mechanisms of source bins are unobserved; this is not a verified human-generated cohort. This comparison matches nominal source lengths: source encoding and normalization are unavailable, so these are not exact matched-length samples. The random curves are calculated distributions, not observed recovery of synthetic passwords and not a cross-corpus operational guessing test. Right: known entropy of four synthetic generators at matching ASCII code-point/byte lengths. Repeating a random two-letter unit preserves 2 log2(26) bits; a random four-letter prefix followed by fixed '1!' and fixed padding preserves 4 log2(26) bits. There are {n_controls:,} generated records across these four categories, 100 per cell and three independent secure-random replicates. Entropy is derived from generator definitions, not estimated from these samples. Fixed padding is a construction example, not an estimate of human behavior."))

    # Figure 5: a preregistered, tiny corpus-membership panel, not a rank estimator.
    transformations = tables["transformations.csv"]
    if len(transformations) != 8:
        raise ValueError("The fixed transformation panel must have exactly eight examples")
    fig, (ax,) = new_figure("Figure 5. Selected transformations remain corpus observations")
    positions = np.arange(len(transformations))
    ax.barh(positions, transformations.corpus_occurrences, color=COLORS[0], height=.64)
    for pos, row in zip(positions, transformations.itertuples()):
        ax.annotate(f"{int(row.corpus_occurrences):,}  (L={row.length})", (row.corpus_occurrences, pos),
                    xytext=(5, 0), textcoords="offset points", fontsize=9, va="center")
    ax.set_yticks(positions, transformations.example)
    ax.invert_yaxis(); ax.set(xscale="log", xlabel="HIBP reported corpus occurrences (log scale)", xlim=(1e4, 4e8))
    format_log_axis(ax, "x"); ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=.15)
    files += save(fig, out, "fig05_fixed_transformations",
                  "Empirical: eight fixed public examples; historical corpus counts. No global ranks, causal effect or representative denominator.")
    captions.append((5, "fig05_fixed_transformations", "Empirical fixed-example observations",
        "Historical Pwned Passwords corpus-occurrence counts for eight selected public examples queried through the k-anonymity range API. Counts are displayed on a logarithmic axis and each label includes observed byte/ASCII character length. The panel's sample size is eight deliberately selected examples, not a random sample of human passwords. Global guessing ranks and a representative population denominator are unavailable. Counts establish corpus presence and relative counts among this panel; they do not prove current use, identify any person, estimate a causal benefit of a transformation or compare passwords against an implemented attack. Query provenance is retained in the dataset audit."))

    # Figure 6: every frozen exploratory model family, with identical folds/target.
    diagnostic = tables["nonlinear_model_validation.csv"]
    held_models = diagnostic[diagnostic.fit_scope == "held_length"]
    model = held_models[held_models.model == "linear_logistic"].sort_values("held_length")
    if model.held_length.tolist() != list(range(6, 17)):
        raise ValueError("All eleven held-length diagnostics are required")
    fig, (ax, bx) = new_figure("Figure 6. Nonlinear length-model diagnostics across held-out groups", 2, 11.8)
    for axis in (ax, bx):
        axis.plot(model.held_length, 100 * model.observed_distinct_early_fraction,
                  "o-", color=COLORS[0], label="Observed distinct-bin fraction")
        axis.set(xlabel="Held-out source-defined length group",
                 ylabel="Distinct Anonymous Bins in pooled top 10,000 (%)", yscale="log")
        axis.set_xticks(range(6, 17, 2)); format_log_axis(axis)
    for name, label, color, style in (
        ("linear_logistic", "Linear logistic", COLORS[1], "--"),
        ("quadratic_logistic", "Quadratic logistic", COLORS[2], "-."),
        ("cubic_logistic", "Cubic logistic", COLORS[3], ":"),
    ):
        data = held_models[held_models.model == name].sort_values("held_length")
        ax.plot(data.held_length, 100 * data.predicted_fraction, style, color=color, label=label)
    for name, label, color, style in (
        ("intercept_only", "Training-group constant", COLORS[1], ":"),
        ("bspline_cubic_4knots", "Cubic B-spline: 4 knots", COLORS[2], "--"),
        ("bspline_cubic_5knots", "Cubic B-spline: 5 knots", COLORS[3], "-."),
    ):
        data = held_models[held_models.model == name].sort_values("held_length")
        bx.plot(data.held_length, 100 * data.predicted_fraction, style, color=color, label=label)
    lower = 100 * min(held_models.predicted_fraction.min(), model.observed_distinct_early_fraction.min()) / 1.5
    upper = 100 * max(held_models.predicted_fraction.max(), model.observed_distinct_early_fraction.max()) * 1.5
    for axis in (ax, bx):
        axis.set_ylim(lower, upper); axis.legend(frameon=False, loc="upper right")
    ax.set_title("Polynomial length terms", loc="left")
    bx.set_title("Spline length terms and constant baseline", loc="left")
    files += save(fig, out, "fig06_length_model_diagnostic",
                  f"Exploratory held-length diagnostics; bins={int(model.test_anonymous_bins.sum()):,}. Human length + structure: DATA UNAVAILABLE.")
    captions.append((6, "fig06_length_model_diagnostic", "Empirical descriptive model diagnostic",
        f"All six fixed exploratory models are evaluated on the same eleven leave-one-length-out folds: intercept-only, linear/quadratic/cubic logistic, and cubic B-splines with four/five uniformly spaced knots. The left panel shows polynomial fits; the right shows spline fits and the constant baseline, with observed fractions repeated for reference. Models are trained on ten released length groups and predict the eleventh. Transformations and knots use only training lengths; spline extrapolation is linear beyond training knots. No held-fold tuning is performed. The outcome is membership among the pooled top 10,000 anonymous bins, with fractional membership at the tied boundary and one weight per distinct bin. The cutoff and outcome labels use the full pool, including the held-out group, so this is not prospective validation. The held-group denominator totals {int(model.test_anonymous_bins.sum()):,} Distinct Anonymous Bins. Vertical values are percentages of distinct bins, not Account Mass coverage. The comparison was added after inspecting the original linear miscalibration and remains exploratory. This exercise diagnoses aggregate interpolation/extrapolation, not independent prediction of individuals or password strings. Human length plus structure remains DATA UNAVAILABLE."))

    # Figure 7: budget/rate arithmetic, not expected time to recover an account.
    fig, (ax, bx) = new_figure("Figure 7. Time to execute a hypothetical fixed checking budget", 2, 11.2)
    budgets = np.logspace(1, 6, 120)
    for i, rate in enumerate((1, 10, 100)):
        ax.plot(budgets, budgets / rate / 3600, color=COLORS[i], linestyle="--", label=f"Assumed {rate} checks/s")
    algorithm_names = {"pbkdf2_sha256": "PBKDF2-SHA256, 600k", "argon2id": "Argon2id, 64 MiB/t3/p1", "bcrypt": "bcrypt, log2 rounds 12"}
    local_latencies = {}
    for i, (algorithm, label) in enumerate(algorithm_names.items()):
        data = hashing[(hashing.algorithm == algorithm) & (hashing.cost_level == 3)]
        latency = float(data.verify_median_ms.median()) / 1000
        local_latencies[algorithm] = latency
        bx.plot(budgets, budgets * latency / 3600, color=COLORS[i], label=label)
    for axis in (ax, bx):
        axis.set(xscale="log", yscale="log", xlabel="Hypothetical checking budget K", ylabel="Time to execute K checks (hours)", ylim=(1e-5, 1e3))
        format_log_axis(axis, "x"); format_log_axis(axis)
        axis.legend(frameon=False, loc="upper left")
    ax.set_title("Online scenario: assumed allowed rates", loc="left")
    bx.set_title("Offline illustration: local serial verifier cost", loc="left")
    files += save(fig, out, "fig07_threat_model_budget_time",
                  "Theoretical time arithmetic. Offline inputs are observed one-process CPU latencies; neither panel predicts password recovery time.")
    captions.append((7, "fig07_threat_model_budget_time", "Mathematical scenarios informed by local measurement",
        "Time to execute a fixed hypothetical budget K is K/r for online rates of 1, 10 and 100 checks/s, or K times local successful-verification latency for the highest configured hashing level. Offline illustration uses the median across four length-cell verification medians: " + "; ".join(f"{algorithm}: {latency * 1000:.3f} ms" for algorithm, latency in local_latencies.items()) + ". These are single-process CPU measurements, not optimized GPU/ASIC attacker throughput, concurrency, universal offline rates or expected time to recover any password. Online rates are hypothetical and omit account-specific controls; rate limiting, detection and MFA can materially change the scenario. No authentication service is contacted. The fixed-budget arithmetic has no empirical population denominator."))

    # Figure 8: all hashing levels and length cells; show IQR instead of fake CIs.
    fig, axes = new_figure("Figure 8. Configured storage cost dominates measured length variation", 3, 12.2)
    for ax, (algorithm, label) in zip(axes, algorithm_names.items()):
        for level in (1, 2, 3):
            data = hashing[(hashing.algorithm == algorithm) & (hashing.cost_level == level)].sort_values("length_bytes")
            color = COLORS[level - 1]
            y = data.verify_median_ms.to_numpy()
            errors = np.vstack((y - data.verify_q25_ms, data.verify_q75_ms - y))
            ax.errorbar(data.length_bytes, y, yerr=errors, marker="o", markersize=4, capsize=2,
                        color=color, label=f"Level {level} verify")
            ax.plot(data.length_bytes, data.hash_median_ms, "x--", color=color, markersize=4, alpha=.75)
        ax.set(xlabel="Synthetic ASCII input length (bytes)", ylabel="Median local latency (ms)", yscale="log", ylim=(8, 300))
        ax.set_xticks([8, 16, 32, 64]); ax.set_yticks([10, 30, 100, 300], ["10", "30", "100", "300"])
        ax.set_title({"pbkdf2_sha256": "PBKDF2-HMAC-SHA256", "argon2id": "Argon2id (p=1)", "bcrypt": "bcrypt"}[algorithm], loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(.065, .915))
    files += save(fig, out, "fig08_hashing_latency",
                  "36 cells × 5 timed repetitions per operation. Solid circles: verify; dashed crosses: hash; bars: IQR, not confidence intervals.")
    captions.append((8, "fig08_hashing_latency", "Empirical controlled local benchmark",
        "All algorithm–cost–length cells of the local benchmark are shown with a shared log-latency scale. Solid lines/circles are successful verification medians; dashed lines/crosses are hashing medians, including secure salt generation. Error bars are verification IQRs from five timed repetitions after one warmup; they are not confidence intervals. PBKDF2 levels use 100k/300k/600k iterations. Argon2id levels use 19/32/64 MiB and passes 2/2/3, always parallelism 1. bcrypt uses log2 rounds 8/10/12; level 1 is a historical comparison. There are 36 cells and 180 timed observations per operation, plus excluded warmups and incorrect-input checks. All inputs are fresh synthetic ASCII strings; bcrypt inputs longer than 72 bytes are rejected and not benchmarked. The 64 MiB Argon2 level differs from RFC 9106's p=4 option. Differences do not rank algorithm security, establish equivalence or estimate attacker throughput. Actual versions and host metadata are in hashing_metadata.json."))

    # Figure 9: observed random-word lengths plus exact conditional tuple counts.
    phrases = tables["passphrase_summary.csv"]
    conditional = tables["passphrase_conditional.csv"]
    fig, (ax, bx) = new_figure("Figure 9. Random-word length and entropy answer different questions", 2, 11.2)
    for i, (word_count, data) in enumerate(phrases.groupby("word_count")):
        color = COLORS[i]
        length = float(data.median_length.median())
        entropy = float(data.entropy_bits.iloc[0])
        low, high = float(data.q10_length.median()), float(data.q90_length.median())
        ax.errorbar(length, entropy, xerr=np.array([[length - low], [high - length]]),
                    marker="o", capsize=4, color=color, label=f"{int(word_count)} words")
        ax.scatter(data.median_length, data.entropy_bits, s=14, color=color, alpha=.6)
        subset = conditional[conditional.word_count == word_count].sort_values("length")
        bx.plot(subset.length, subset.conditional_entropy_bits, color=color, label=f"{int(word_count)} words")
        bx.hlines(entropy, subset.length.min(), subset.length.max(), color=color, linestyles="--", alpha=.45)
    ax.set(xlabel="Observed phrase length (characters, including spaces)", ylabel="Unconditional generator entropy (bits)")
    bx.set(xlabel="Known total character length (spaces included)", ylabel="Exact conditional word-tuple entropy (bits)")
    ax.legend(frameon=False, loc="upper left"); bx.legend(frameon=False, loc="upper left")
    ax.set_title("Observed length ranges; entropy is mathematical", loc="left")
    bx.set_title("Conditioning on length narrows tuple support", loc="left")
    phrase_n = int(phrases.n.sum())
    files += save(fig, out, "fig09_random_word_passphrases",
                  f"EFF vocabulary D=7,776; {phrase_n:,} securely sampled phrases. Bars describe sampled length quantiles; dashed lines show unconditional entropy.")
    captions.append((9, "fig09_random_word_passphrases", "Empirical synthetic lengths and theoretical entropy",
        f"Left: {phrase_n:,} synthetic phrases from uniform independent sampling with replacement from the 7,776-word EFF vocabulary, 100 phrases per word-count cell in each of three independent CSPRNG replicates. Individual dots are replicate median lengths; the central marker is their median; horizontal bars are medians of the replicate 10th–90th length quantiles and are descriptive, not confidence intervals. Words are joined with one space, including spaces in character length, so hyphen-containing vocabulary entries remain unambiguous. The vertical coordinate w log2(7776) is a theoretical generator property. Right: log2 of exact counts of word tuples with each total length, computed by convolution of vocabulary word-length multiplicities, conditional on known word count and length. Dashed lines show unconditional entropy. Theoretical rare lengths are retained; no human-sentence model, observed guessing resistance or empirical entropy estimate is asserted."))

    caption_lines = ["# Figure captions", "", "Each figure is saved as 300 dpi PNG, PDF and editable SVG. Titles and footers are part of the export; captions provide the full interpretation and denominator.", ""]
    for number, stem, result_type, text in captions:
        caption_lines.extend([f"## Figure {number}", "", f"**Evidence type:** {result_type}.", "", text, "", f"Files: [{stem}.png]({stem}.png), [{stem}.pdf]({stem}.pdf), [{stem}.svg]({stem}.svg).", ""])
    (out / "captions.md").write_text("\n".join(caption_lines), encoding="utf-8")
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(Path(__file__).resolve()),
        "input_sha256": {name: sha(table_dir / name) for name in INPUT_FILES},
        "output_sha256": {name: sha(out / name) for name in files},
        "captions_sha256": sha(out / "captions.md"),
        "matplotlib_version": matplotlib.__version__, "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "figures": len(captions), "files": files,
        "notes": "Figures read saved aggregate tables only; no password strings except eight explicitly public illustrative labels",
    }
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(captions), "exports": len(files), "directory": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.root.resolve())
