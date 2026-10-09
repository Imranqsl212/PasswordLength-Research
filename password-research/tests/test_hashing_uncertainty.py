"""Offline mathematical/provenance checks; no timing benchmark is executed."""

import csv
import hashlib
import importlib.util
import itertools
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/hashing/derive_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("hashing_uncertainty", MODULE_PATH)
uncertainty = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = uncertainty
SPEC.loader.exec_module(uncertainty)


def fixture(values):
    ordered = sorted(values)
    return {
        "algorithm": "fixture", "cost_level": "1", "parameters": "{}",
        "length_bytes": "8", "length_characters": "8", "repetitions": "5",
        "warmups": "1",
        **{f"hash_{name}_ms": repr(value) for name, value
           in zip(uncertainty.ORDER_STATISTICS, ordered)},
    }


class HashingUncertaintyTests(unittest.TestCase):
    def test_every_ordering_recovers_same_sample_variance(self):
        # Includes repeated values and an outlier; order is irrelevant to variance.
        for values in itertools.permutations((1.25, 1.25, 2.5, 3.0, 200.75)):
            row = uncertainty.derive_row(fixture(values), "hash")
            self.assertEqual(row["sample_variance_ms2"], statistics.variance(values))
            self.assertEqual(row["mean_ms"], statistics.mean(values))

    def test_hand_calculation_units_and_ddof(self):
        row = uncertainty.derive_row(fixture((1, 2, 3, 4, 5)), "hash")
        self.assertEqual(row["median_ms"], 3)
        self.assertEqual(row["iqr_ms"], 2)
        self.assertEqual(row["sample_variance_ms2"], 2.5)
        self.assertAlmostEqual(row["sample_sd_ms"], 2.5 ** .5)
        self.assertAlmostEqual(row["coefficient_of_variation"], (2.5 ** .5) / 3)

    def test_no_general_reconstruction_for_other_sample_counts(self):
        for n in (1, 2, 4, 6, 10):
            row = fixture((1, 2, 3, 4, 5)); row["repetitions"] = str(n)
            with self.assertRaises(ValueError):
                uncertainty.reconstruct_sorted(row, "hash")

    def test_invalid_summary_is_rejected(self):
        for key, value in (("hash_q25_ms", "10"), ("hash_min_ms", "-1"),
                           ("hash_max_ms", "nan"), ("hash_max_ms", "inf")):
            row = fixture((1, 2, 3, 4, 5)); row[key] = value
            with self.assertRaises(ValueError):
                uncertainty.reconstruct_sorted(row, "hash")
        with self.assertRaises(ValueError):
            uncertainty.reconstruct_sorted(fixture((1, 2, 3, 4, 5)), "wrong")

    def test_zero_timings_do_not_invent_coefficient_of_variation(self):
        row = uncertainty.derive_row(fixture((0, 0, 0, 0, 0)), "hash")
        self.assertEqual(row["sample_variance_ms2"], 0)
        self.assertEqual(row["coefficient_of_variation"], "")

    def test_frozen_source_integrity_and_extreme_observation_retained(self):
        rows, metadata = uncertainty.load_frozen(ROOT)
        self.assertEqual(len(rows), 36)
        self.assertEqual(metadata["configuration"]["repetitions"], 5)
        row = next(r for r in rows if r["algorithm"] == "argon2id"
                   and r["cost_level"] == "2" and r["length_bytes"] == "8")
        result = uncertainty.derive_row(row, "hash")
        self.assertGreater(result["max_ms"], 299)
        self.assertGreater(result["sample_variance_ms2"], 13000)
        self.assertLess(result["iqr_ms"], 1)

    def test_tampered_frozen_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results/tables").mkdir(parents=True)
            (root / "results/statistics").mkdir(parents=True)
            for name in ("results/tables/hashing_cost.csv", "results/statistics/hashing_metadata.json"):
                (root / name).write_bytes((ROOT / name).read_bytes())
            with (root / "results/tables/hashing_cost.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "timing table SHA256"):
                uncertainty.load_frozen(root)

    def test_export_replay_is_deterministic_and_preserves_originals(self):
        originals = (ROOT / "results/tables/hashing_cost.csv",
                     ROOT / "results/statistics/hashing_metadata.json")
        before = [path.read_bytes() for path in originals]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("results/tables/hashing_cost.csv", "results/statistics/hashing_metadata.json",
                         "experiments/hashing/benchmark.py"):
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            table, metadata_path = uncertainty.derive(root)
            first = table.read_bytes()
            metadata = json.loads(metadata_path.read_text())
            uncertainty.derive(root)
            self.assertEqual(table.read_bytes(), first)
            self.assertEqual(metadata["output_sha256"], hashlib.sha256(first).hexdigest())
            self.assertEqual(metadata["new_timed_operations"], 0)
            with table.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 72)
        self.assertEqual([path.read_bytes() for path in originals], before)


if __name__ == "__main__":
    unittest.main()
