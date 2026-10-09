#!/usr/bin/env python3
"""Derive descriptive uncertainty from the frozen five-observation timing run.

For n=5, the original linear-interpolation quartiles and endpoints preserve
all five sorted observations. This script does no hashing or verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ORDER_STATISTICS = ("min", "q25", "median", "q75", "max")
METHOD = {
    "required_repetitions": 5,
    "original_quantile_position": "(n-1)*q; q=0,.25,.5,.75,1",
    "sorted_observation_reconstruction": "min,q25,median,q75,max",
    "variance": "sum((x-mean(x))**2)/(n-1); ddof=1; units ms^2",
    "standard_deviation": "sqrt(sample_variance_ms2); units ms",
    "coefficient_of_variation": "sample_sd_ms/mean_ms; unitless",
    "iqr": "q75_ms-q25_ms; descriptive, not a confidence interval",
    "interpretation": "Within-cell observed variability; no population inference",
    "excluded_operations": "Original warmups and untimed incorrect-input checks",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_sorted(row: dict[str, str], operation: str) -> list[float]:
    if operation not in ("hash", "verify"):
        raise ValueError("Operation must be hash or verify")
    if int(row["repetitions"]) != 5:
        raise ValueError("Exactly five repetitions are necessary for this reconstruction")
    values = [float(row[f"{operation}_{name}_ms"]) for name in ORDER_STATISTICS]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timings must be finite and nonnegative")
    if values != sorted(values):
        raise ValueError("Saved order statistics must be nondecreasing")
    return values


def derive_row(row: dict[str, str], operation: str) -> dict[str, object]:
    values = reconstruct_sorted(row, operation)
    mean = statistics.mean(values)
    variance = statistics.variance(values)
    sd = math.sqrt(variance)
    return {
        "algorithm": row["algorithm"],
        "cost_level": int(row["cost_level"]),
        "parameters": row["parameters"],
        "length_bytes": int(row["length_bytes"]),
        "length_characters": int(row["length_characters"]),
        "operation": operation,
        "repetitions": 5,
        "warmups": int(row["warmups"]),
        **{f"{name}_ms": value for name, value in zip(ORDER_STATISTICS, values)},
        "mean_ms": mean,
        "iqr_ms": values[3] - values[1],
        "sample_variance_ms2": variance,
        "sample_sd_ms": sd,
        "coefficient_of_variation": sd / mean if mean else "",
    }


def load_frozen(root: Path) -> tuple[list[dict[str, str]], dict]:
    table = root / "results/tables/hashing_cost.csv"
    metadata_path = root / "results/statistics/hashing_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "COMPLETED":
        raise ValueError("A completed frozen run is required")
    if metadata["configuration"]["repetitions"] != 5:
        raise ValueError("The frozen run must use exactly five repetitions")
    if "Linear interpolation" not in metadata["quantiles"]:
        raise ValueError("The saved quantile convention is unsupported")
    if digest(table) != metadata["output_sha256"]:
        raise ValueError("Frozen timing table SHA256 differs from its run metadata")
    source = root / metadata["source_relative_path"]
    if digest(source) != metadata["source_sha256"]:
        raise ValueError("Original benchmark source SHA256 differs from run metadata")
    config_hash = hashlib.sha256(json.dumps(metadata["configuration"], sort_keys=True)
                                 .encode()).hexdigest()
    if config_hash != metadata["configuration_sha256"]:
        raise ValueError("Frozen run configuration SHA256 is inconsistent")
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != metadata["cells_completed"]:
        raise ValueError("Frozen cell count differs from run metadata")
    keys = [(r["algorithm"], r["cost_level"], r["length_bytes"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Frozen timing cell identifiers are duplicated")
    for row in rows:
        if int(row["warmups"]) != metadata["configuration"]["warmups_per_cell"]:
            raise ValueError("Frozen warmup count differs from configuration")
        for operation in ("hash", "verify"):
            reconstruct_sorted(row, operation)
    return rows, metadata


def derive(root: Path) -> tuple[Path, Path]:
    rows, original_metadata = load_frozen(root)
    derived = [derive_row(row, operation) for row in rows
               for operation in ("hash", "verify")]
    output = root / "results/tables/hashing_uncertainty.csv"
    metadata_path = root / "results/statistics/hashing_uncertainty_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(derived[0]))
        writer.writeheader()
        writer.writerows(derived)
    host_path = root / "results/statistics/hashing_host_observation.json"
    inputs = ["results/tables/hashing_cost.csv", "results/statistics/hashing_metadata.json"]
    if host_path.exists():
        inputs.append("results/statistics/hashing_host_observation.json")
    metadata = {
        "status": "COMPLETED",
        "result_type": "Derived descriptive statistics from frozen empirical timings",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_run_started_at_utc": original_metadata["started_at_utc"],
        "input_sha256": {name: digest(root / name) for name in inputs},
        "original_benchmark_source_sha256": original_metadata["source_sha256"],
        "original_configuration_sha256": original_metadata["configuration_sha256"],
        "derivation_source_relative_path": "experiments/hashing/derive_uncertainty.py",
        "derivation_source_sha256": digest(Path(__file__).resolve()),
        "method": METHOD,
        "method_sha256": hashlib.sha256(json.dumps(METHOD, sort_keys=True).encode()).hexdigest(),
        "output_relative_path": "results/tables/hashing_uncertainty.csv",
        "output_sha256": digest(output),
        "rows": len(derived),
        "cells": len(rows),
        "new_timed_operations": 0,
        "lossless_for": "Sorted five-value multisets at saved IEEE754 float precision",
        "unavailable": [
            "Timing observation order within each cell",
            "Hash/verification pairing, covariance, temporal drift and autocorrelation",
            "Untimed negative-input verification latency",
            "Historical exact CPU model, RAM/core counts, power/thermal/background state",
            "Population timing variance or validated confidence intervals",
            "Measured sustained throughput, concurrent or GPU/ASIC results",
        ],
        "hardware_provenance": "Optional host observation is later context; it does not retroactively fill historical run metadata",
        "original_outputs": "Preserved byte-for-byte; no new secrets, salts or verifier records generated",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    table, metadata = derive(args.root.resolve())
    print(json.dumps({"status": "COMPLETED", "table": str(table),
                      "metadata": str(metadata)}))


if __name__ == "__main__":
    main()
