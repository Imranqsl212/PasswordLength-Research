"""Small correctness tests; all secrets below are synthetic test fixtures."""

import importlib.util
import csv
import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments/hashing/benchmark.py"
SPEC = importlib.util.spec_from_file_location("hashing_benchmark", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class HashingTests(unittest.TestCase):
    def test_synthetic_lengths_are_ascii_byte_lengths(self):
        inputs = benchmark.synthetic_inputs(2, 1)
        self.assertEqual(set(inputs), {8, 16, 32, 64})
        for length, group in inputs.items():
            self.assertEqual(len(group), 3)
            for secret in group:
                self.assertEqual(len(secret), length)
                self.assertEqual(len(secret.decode("ascii")), length)
                self.assertTrue(set(secret.decode("ascii")) <= set(benchmark.ASCII_ALPHABET))

    def test_randomization_is_explicitly_bounded(self):
        for repetitions, warmups in ((0, 1), (11, 1), (5, -1), (5, 4)):
            with self.assertRaises(ValueError):
                benchmark.synthetic_inputs(repetitions, warmups)

    def test_pbkdf2_correctness_salt_and_wrong_input(self):
        adapter = benchmark.pbkdf2_adapter(10, 1)
        first = adapter.hash_secret(b"synthetic fixture")
        second = adapter.hash_secret(b"synthetic fixture")
        self.assertTrue(adapter.verify_secret(b"synthetic fixture", first))
        self.assertFalse(adapter.verify_secret(b"different fixture", first))
        self.assertEqual(len(first.salt), 16)
        self.assertEqual(len(first.digest), 32)
        self.assertNotEqual(first.salt, second.salt)
        self.assertNotEqual(first.digest, second.digest)

    def test_quantiles(self):
        values = [5.0, 1.0, 4.0, 2.0, 3.0]
        result = benchmark.summary(values, "hash")
        self.assertEqual(result["hash_median_ms"], 3)
        self.assertEqual(result["hash_q25_ms"], 2)
        self.assertEqual(result["hash_q75_ms"], 4)
        self.assertEqual(benchmark.linear_quantile([1.0, 3.0], .25), 1.5)
        with self.assertRaises(ValueError):
            benchmark.summary([], "hash")

    def test_input_bundle_digest_is_framed_and_sensitive(self):
        first = {8: [b"abcdefgh", b"ABCDEFGH"]}
        second = {8: [b"abcdefgh", b"ABCDEFGI"]}
        self.assertEqual(benchmark.input_digest(first), benchmark.input_digest(first))
        self.assertNotEqual(benchmark.input_digest(first), benchmark.input_digest(second))
        self.assertEqual(len(benchmark.input_digest(first)), 64)

    def test_timing_rows_export_aggregates_only(self):
        adapter = benchmark.pbkdf2_adapter(10, 1)
        inputs = [b"fixture1", b"fixture2", b"fixture3"]
        row = benchmark.measure_cell(adapter, inputs, 1, time.perf_counter() + 5)
        self.assertEqual(row["repetitions"], 2)
        self.assertEqual(row["correct_verification_checks"], 3)
        self.assertEqual(row["incorrect_verification_checks"], 1)
        serialized = json.dumps(row)
        self.assertNotIn("fixture1", serialized)
        self.assertNotIn("digest", row)
        self.assertNotIn("salt", row)
        self.assertNotIn("secret", row)
        self.assertNotIn("record", row)

    def test_deadline_and_cost_bounds(self):
        adapter = benchmark.pbkdf2_adapter(10, 1)
        with self.assertRaises(TimeoutError):
            benchmark.measure_cell(adapter, [b"fixture1"], 0, time.perf_counter() - 1)
        with self.assertRaises(ValueError):
            benchmark.pbkdf2_adapter(600_001, 1)

    def test_export_metadata_matches_table_and_operation_counts(self):
        # A cheap isolated export smoke test, not another full benchmark.
        adapter = benchmark.pbkdf2_adapter(10, 1)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(benchmark, "build_adapters", return_value=([adapter], [])):
                table, metadata_path = benchmark.run_benchmark(Path(directory), 2, 1, 5)
            with table.open() as handle:
                rows = list(csv.DictReader(handle))
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(len(rows), 4)
            self.assertEqual(metadata["cells_completed"], 4)
            self.assertEqual(metadata["hash_operations"], 12)
            self.assertEqual(metadata["positive_verify_operations"], 12)
            self.assertEqual(metadata["negative_verify_operations"], 4)
            self.assertEqual(metadata["output_sha256"], hashlib.sha256(table.read_bytes()).hexdigest())
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest())
            self.assertEqual(metadata["status"], "COMPLETED")
            self.assertFalse(set(rows[0]) & {"secret", "password", "salt", "digest", "verifier"})

    def test_argon2id_correctness_if_available(self):
        try:
            adapter = benchmark.argon2_adapter(32, 1, 1)
        except ImportError:
            self.skipTest("argon2-cffi unavailable")
        first = adapter.hash_secret(b"synthetic fixture")
        second = adapter.hash_secret(b"synthetic fixture")
        self.assertTrue(first.startswith("$argon2id$v=19$"))
        self.assertTrue(adapter.verify_secret(b"synthetic fixture", first))
        self.assertFalse(adapter.verify_secret(b"different fixture", first))
        self.assertNotEqual(first, second)

    def test_bcrypt_correctness_and_byte_limit_if_available(self):
        try:
            adapter = benchmark.bcrypt_adapter(4, 1)
        except ImportError:
            self.skipTest("bcrypt unavailable")
        first = adapter.hash_secret(b"synthetic fixture")
        self.assertTrue(adapter.verify_secret(b"synthetic fixture", first))
        self.assertFalse(adapter.verify_secret(b"different fixture", first))
        self.assertTrue(adapter.verify_secret(b"a" * 72, adapter.hash_secret(b"a" * 72)))
        with self.assertRaises(ValueError):
            adapter.hash_secret(b"a" * 73)
        with self.assertRaises(ValueError):
            adapter.hash_secret(("é" * 37).encode("utf-8"))
        with self.assertRaises(ValueError):
            adapter.verify_secret(b"a" * 73, first)


if __name__ == "__main__":
    unittest.main()
