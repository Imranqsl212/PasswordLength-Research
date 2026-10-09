"""Count-only research utilities. No candidate enumeration or account interaction."""
from collections import Counter
import math
import re
import string
import numpy as np

ALPHABETS = {
    "lowercase26": string.ascii_lowercase,
    "letters52": string.ascii_letters,
    "alphanumeric62": string.ascii_letters + string.digits,
    "printable94": string.ascii_letters + string.digits + string.punctuation,
}

def parse_histogram(text):
    counts = Counter()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"line {number}: expected frequency multiplicity")
        f, m = map(int, parts)
        if f <= 0 or m <= 0:
            raise ValueError(f"line {number}: counts must be positive")
        counts[f] += m
    if not counts:
        raise ValueError("empty histogram")
    return sorted(counts.items(), reverse=True)

def mass(hist):
    return sum(f * m for f, m in hist)

def size(hist):
    return sum(m for _, m in hist)

def coverage(hist, budget):
    if budget < 0:
        raise ValueError("negative budget")
    left, covered = budget, 0
    for f, m in hist:
        take = min(left, m)
        covered += f * take
        left -= take
        if left <= 0:
            break
    return covered / mass(hist)

def guesses_for_mass(hist, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("fraction outside (0,1]")
    target, covered, rank = fraction * mass(hist), 0, 0
    for f, m in hist:
        if covered + f * m >= target:
            return rank + math.ceil((target - covered) / f)
        covered += f * m
        rank += m
    return rank

def rank_map(hist):
    result, before = {}, 0
    for f, m in hist:
        result[f] = (before + 1, before + m, before + (m + 1) / 2)
        before += m
    return result

def pooled_tie_coverage(hist, pooled, budget):
    """Expected account/bin coverage and sharp allocation bounds within a tie."""
    ranks = rank_map(pooled)
    pool_counts = dict(pooled)
    expected = lower = upper = distinct = 0.0
    for f, m in hist:
        start, end, _ = ranks[f]
        slots = max(0, min(pool_counts[f], budget - start + 1))
        take = slots * m / pool_counts[f]
        lo = max(0, slots - (pool_counts[f] - m))
        hi = min(m, slots)
        expected += f * take
        lower += f * lo
        upper += f * hi
        distinct += take
    return expected / mass(hist), lower / mass(hist), upper / mass(hist), distinct / size(hist)

def weighted_quantile(values, weights, q):
    if not 0 <= q <= 1:
        raise ValueError("invalid quantile")
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    if weights.sum() <= 0 or (weights < 0).any():
        raise ValueError("invalid weights")
    i = np.searchsorted(np.cumsum(weights), q * weights.sum(), side="left")
    return float(values[min(i, len(values)-1)])

def weighted_midranks(values, weights):
    values, weights = np.asarray(values), np.asarray(weights, dtype=float)
    _, inverse = np.unique(values, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    midranks = np.cumsum(totals) - totals + (totals + 1) / 2
    return midranks[inverse]

def weighted_spearman(x, y, weights):
    w = np.asarray(weights, dtype=float)
    a, b = weighted_midranks(x, w), weighted_midranks(y, w)
    a -= np.average(a, weights=w)
    b -= np.average(b, weights=w)
    den = math.sqrt(np.sum(w*a*a) * np.sum(w*b*b))
    return float(np.sum(w*a*b) / den) if den else None

def uniform_baseline(alphabet_size, length):
    if alphabet_size < 2 or length < 1:
        raise ValueError("invalid alphabet or length")
    space = alphabet_size ** length
    return {"space": space, "expected_guesses": (space+1)/2,
            "entropy_bits": length * math.log2(alphabet_size)}

def uniform_coverage(alphabet_size, length, budget):
    return min(1.0, max(0, budget) / alphabet_size ** length)

def features(value):
    """Literal code-point features; no Unicode, case, or whitespace normalization."""
    n = len(value)
    unique = len(set(value))
    cls = lambda c: "L" if c.islower() else "U" if c.isupper() else "D" if c.isdigit() else "W" if c.isspace() else "A" if c.isalnum() else "S"
    classes = "".join(map(cls, value))
    repeated = any(value == value[:k] * (n // k) for k in range(1, n//2+1) if n % k == 0)
    keyboard = any(x in value.lower() for x in ("qwerty", "asdf", "zxcv", "123456"))
    runs = lambda pattern: max((len(x) for x in re.findall(pattern, value)), default=0)
    return {"length":n, "utf8_bytes":len(value.encode("utf-8")), "unique_characters":unique,
            "diversity":unique/n if n else 0,
            "digits":sum(c.isdigit() for c in value), "uppercase":sum(c.isupper() for c in value),
            "lowercase":sum(c.islower() for c in value), "whitespace":sum(c.isspace() for c in value),
            "symbols":sum(not c.isalnum() and not c.isspace() for c in value),
            "starts_digit":int(bool(value) and value[0].isdigit()), "ends_digit":int(bool(value) and value[-1].isdigit()),
            "starts_uppercase":int(bool(value) and value[0].isupper()), "ends_symbol":int(bool(value) and cls(value[-1])=="S"),
            "adjacent_repeat":int(any(a==b for a,b in zip(value,value[1:]))),
            "whole_string_repeated_substring":int(repeated),
            "max_ascii_digit_run":runs(r"[0-9]+"), "max_ascii_letter_run":runs(r"[a-zA-Z]+"),
            "alternating_character_classes":int(n>=4 and classes[:2]* (n//2)+classes[:n%2]==classes and classes[0]!=classes[1]),
            "year_1900_2099":int(bool(re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)",value))),
            "literal_keyboard_fragment":int(keyboard)}
