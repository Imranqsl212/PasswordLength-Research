#!/usr/bin/env python3
"""Bounded, single-process password-storage latency experiment.

Only fresh synthetic secrets are accepted as experimental inputs. No password,
salt, or verifier is written to disk; outputs contain aggregate timings only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import importlib.metadata
import json
import platform
import random
import secrets
import statistics
import string
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


LENGTHS = (8, 16, 32, 64)
ORDER_SEED = 20261005
ASCII_ALPHABET = string.ascii_letters + string.digits
PBKDF2_ITERATIONS = (100_000, 300_000, 600_000)
ARGON2_SETTINGS = ((19 * 1024, 2), (32 * 1024, 2), (64 * 1024, 3))
BCRYPT_ROUNDS = (8, 10, 12)


@dataclass(repr=False, frozen=True)
class PBKDF2Record:
    salt: bytes = field(repr=False)
    digest: bytes = field(repr=False)


@dataclass
class Adapter:
    algorithm: str
    level: int
    parameters: dict[str, Any]
    hash_secret: Callable[[bytes], Any] = field(repr=False)
    verify_secret: Callable[[bytes, Any], bool] = field(repr=False)
    library_version: str


def validate_secret(secret: bytes, maximum_bytes: int | None = None) -> None:
    if not isinstance(secret, bytes) or not secret:
        raise ValueError("Expected a nonempty synthetic byte string")
    if maximum_bytes is not None and len(secret) > maximum_bytes:
        raise ValueError(f"Input exceeds the documented {maximum_bytes}-byte limit")


def pbkdf2_adapter(iterations: int, level: int) -> Adapter:
    if not 1 <= iterations <= 600_000:
        raise ValueError("PBKDF2 iteration count exceeds the bounded experiment")

    def hash_secret(secret: bytes) -> PBKDF2Record:
        validate_secret(secret)
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", secret, salt, iterations, dklen=32)
        return PBKDF2Record(salt, digest)

    def verify_secret(secret: bytes, record: PBKDF2Record) -> bool:
        validate_secret(secret)
        digest = hashlib.pbkdf2_hmac("sha256", secret, record.salt, iterations, dklen=32)
        return hmac.compare_digest(digest, record.digest)

    return Adapter(
        "pbkdf2_sha256", level,
        {"iterations": iterations, "salt_bytes": 16, "derived_key_bytes": 32},
        hash_secret, verify_secret, platform.python_version(),
    )


def argon2_adapter(memory_kib: int, time_cost: int, level: int) -> Adapter:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError
    from argon2.low_level import Type

    if not 8 <= memory_kib <= 65_536 or not 1 <= time_cost <= 3:
        raise ValueError("Argon2 parameters exceed the bounded experiment")
    hasher = PasswordHasher(
        time_cost=time_cost, memory_cost=memory_kib, parallelism=1,
        hash_len=32, salt_len=16, type=Type.ID,
    )

    def hash_secret(secret: bytes) -> str:
        validate_secret(secret)
        # The library generates a fresh cryptographically secure salt.
        return hasher.hash(secret)

    def verify_secret(secret: bytes, record: str) -> bool:
        validate_secret(secret)
        try:
            return hasher.verify(record, secret)
        except VerifyMismatchError:
            return False

    return Adapter(
        "argon2id", level,
        {"memory_kib": memory_kib, "time_cost": time_cost, "parallelism": 1,
         "salt_bytes": 16, "derived_key_bytes": 32, "version": 19},
        hash_secret, verify_secret, importlib.metadata.version("argon2-cffi"),
    )


def bcrypt_adapter(rounds: int, level: int) -> Adapter:
    import bcrypt

    if not 4 <= rounds <= 12:
        raise ValueError("bcrypt rounds exceed the bounded experiment")

    def hash_secret(secret: bytes) -> bytes:
        validate_secret(secret, maximum_bytes=72)
        return bcrypt.hashpw(secret, bcrypt.gensalt(rounds=rounds))

    def verify_secret(secret: bytes, record: bytes) -> bool:
        # Guard explicitly, including on versions that silently truncate.
        validate_secret(secret, maximum_bytes=72)
        return bcrypt.checkpw(secret, record)

    return Adapter(
        "bcrypt", level, {"log2_rounds": rounds, "maximum_input_bytes": 72,
                          "salt_bytes": 16, "prefix": "2b"},
        hash_secret, verify_secret, importlib.metadata.version("bcrypt"),
    )


def build_adapters() -> tuple[list[Adapter], list[dict[str, str]]]:
    adapters = [pbkdf2_adapter(cost, level)
                for level, cost in enumerate(PBKDF2_ITERATIONS, 1)]
    unavailable: list[dict[str, str]] = []
    for name, factory, settings in (
        ("argon2id", argon2_adapter, ARGON2_SETTINGS),
        ("bcrypt", bcrypt_adapter, tuple((r,) for r in BCRYPT_ROUNDS)),
    ):
        try:
            group = [factory(*params, level=level)
                     for level, params in enumerate(settings, 1)]
        except ImportError:
            unavailable.append({"algorithm": name, "status": "NOT RUN",
                                "reason": "Optional dependency is unavailable"})
        else:
            adapters.extend(group)
    return adapters, unavailable


def synthetic_inputs(repetitions: int, warmups: int) -> dict[int, list[bytes]]:
    if not 1 <= repetitions <= 10 or not 0 <= warmups <= 3:
        raise ValueError("Use 1–10 repetitions and 0–3 warmups")
    return {
        length: ["".join(secrets.choice(ASCII_ALPHABET) for _ in range(length))
                 .encode("ascii") for _ in range(repetitions + warmups)]
        for length in LENGTHS
    }


def input_digest(inputs: dict[int, list[bytes]]) -> str:
    """Hash a framed synthetic input bundle; preserve no individual secret."""
    digest = hashlib.sha256()
    for length in sorted(inputs):
        digest.update(length.to_bytes(4, "big"))
        digest.update(len(inputs[length]).to_bytes(4, "big"))
        for secret in inputs[length]:
            digest.update(len(secret).to_bytes(4, "big"))
            digest.update(secret)
    return digest.hexdigest()


def linear_quantile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("Quantile requires values and a probability in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values: list[float], prefix: str) -> dict[str, float]:
    if not values or any(value < 0 for value in values):
        raise ValueError("Timing observations must be nonnegative and nonempty")
    return {
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_q25_ms": linear_quantile(values, .25),
        f"{prefix}_q75_ms": linear_quantile(values, .75),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
    }


def measure_cell(adapter: Adapter, inputs: list[bytes], warmups: int,
                 deadline: float) -> dict[str, Any]:
    hash_times: list[float] = []
    verify_times: list[float] = []
    for index, secret in enumerate(inputs):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Experiment exceeded its bounded runtime")
        started = time.perf_counter()
        record = adapter.hash_secret(secret)
        hash_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        matched = adapter.verify_secret(secret, record)
        verify_ms = (time.perf_counter() - started) * 1000
        if not matched:
            raise AssertionError("Correct synthetic secret failed verification")
        if index == 0:
            # Wrong-input correctness check is deliberately outside timing.
            replacement = b"A" if secret[0:1] != b"A" else b"B"
            wrong = replacement + secret[1:]
            if adapter.verify_secret(wrong, record):
                raise AssertionError("Incorrect synthetic secret passed verification")
        if index >= warmups:
            hash_times.append(hash_ms)
            verify_times.append(verify_ms)
        del record
    return {
        "algorithm": adapter.algorithm,
        "cost_level": adapter.level,
        "parameters": json.dumps(adapter.parameters, sort_keys=True),
        "length_bytes": len(inputs[0]),
        "length_characters": len(inputs[0]),
        "repetitions": len(hash_times),
        "warmups": warmups,
        "correct_verification_checks": len(inputs),
        "incorrect_verification_checks": 1,
        **summary(hash_times, "hash"),
        **summary(verify_times, "verify"),
    }


def run_benchmark(root: Path, repetitions: int = 5, warmups: int = 1,
                  max_seconds: float = 180) -> tuple[Path, Path]:
    if not 1 <= max_seconds <= 300:
        raise ValueError("Runtime budget must be within 1–300 seconds")
    adapters, unavailable = build_adapters()
    inputs = synthetic_inputs(repetitions, warmups)
    source = Path(__file__).resolve()
    config = {
        "lengths_bytes": list(LENGTHS), "alphabet": ASCII_ALPHABET,
        "input_generation": "secrets.choice; fresh synthetic printable ASCII",
        "repetitions": repetitions, "warmups_per_cell": warmups,
        "order_seed": ORDER_SEED, "max_seconds": max_seconds,
        "algorithms": [{"algorithm": a.algorithm, "cost_level": a.level,
                        "parameters": a.parameters,
                        "library_version": a.library_version} for a in adapters],
    }
    cells = [(adapter, length) for adapter in adapters for length in LENGTHS]
    random.Random(ORDER_SEED).shuffle(cells)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    deadline = started + max_seconds
    rows: list[dict[str, Any]] = []
    for adapter, length in cells:
        rows.append(measure_cell(adapter, inputs[length], warmups, deadline))
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: (row["algorithm"], row["cost_level"], row["length_bytes"]))
    table = root / "results" / "tables" / "hashing_cost.csv"
    metadata_path = root / "results" / "statistics" / "hashing_metadata.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "COMPLETED", "result_type": "Empirical local benchmark",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed, "configuration": config,
        "configuration_sha256": hashlib.sha256(json.dumps(config, sort_keys=True)
                                                .encode()).hexdigest(),
        "synthetic_input_bundle_sha256": input_digest(inputs),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_relative_path": "experiments/hashing/benchmark.py",
        "output_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "host": {"system": platform.system(), "release": platform.release(),
                 "machine": platform.machine(), "processor": platform.processor(),
                 "python": sys.version, "openssl": __import__("ssl").OPENSSL_VERSION},
        "clock": {"name": "perf_counter", "monotonic": True,
                  "resolution_seconds": time.get_clock_info("perf_counter").resolution},
        "cells_completed": len(rows), "not_run": unavailable,
        "hash_operations": len(rows) * (repetitions + warmups),
        "positive_verify_operations": len(rows) * (repetitions + warmups),
        "negative_verify_operations": len(rows),
        "input_handling": "Secrets and verifier records remained in process memory; neither exported",
        "timing_scope": "Hash includes salt generation; verify includes local checking; warmups and wrong-input checks excluded from summaries",
        "quantiles": "Linear interpolation on the ordered five measurements; IQR is descriptive variation, not a confidence interval",
        "limitations": [
            "Single host and process; timing is not an attacker throughput estimate",
            "Fresh secrets and salts prevent bitwise reproduction; configuration and method reproduce",
            "ASCII length equals bytes; Unicode and lengths above 64 bytes were not benchmarked",
            "No external login requests, real credentials, candidate enumeration, or offline cracking",
            "Cost levels are experimental settings; low levels are not deployment recommendations",
            "All Argon2id levels use parallelism=1; 64 MiB/t=3 differs from RFC 9106 parallelism=4",
            "bcrypt inputs above 72 bytes are explicitly rejected and not measured",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return table, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    table, metadata = run_benchmark(args.root.resolve(), args.repetitions,
                                    args.warmups, args.max_seconds)
    print(json.dumps({"status": "COMPLETED", "table": str(table),
                      "metadata": str(metadata)}))


if __name__ == "__main__":
    main()
