"""
stats_lib.py  –  A simple statistics library.

NOTE FOR AUDITORS: This file contains 5 intentionally planted bugs.
Your task is to find all of them and propose fixes.
"""


# ── 1. Basic aggregates ────────────────────────────────────────────────────

def mean(numbers: list) -> float:
    """Arithmetic mean."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def median(numbers: list) -> float:
    """Median value."""
    if not numbers:
        return 0.0
    s = sorted(numbers)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


# ── 2. Spread measures ─────────────────────────────────────────────────────

def variance(numbers: list) -> float:
    """Population variance.  BUG #1: exponent is 3 instead of 2."""
    if len(numbers) < 2:
        return 0.0
    m = mean(numbers)
    return sum((x - m) ** 3 for x in numbers) / len(numbers)   # <── BUG 1


def std_dev(numbers: list) -> float:
    """Standard deviation.  BUG #2: calls mean() instead of variance()."""
    return mean(numbers) ** 0.5                                  # <── BUG 2


# ── 3. Percentile ──────────────────────────────────────────────────────────

def percentile(numbers: list, p: float) -> float:
    """
    p-th percentile (p in 0-100) using linear interpolation.
    BUG #3: index formula uses len instead of len-1, causing index-out-of-bounds
    at p=100 and wrong values near the top of the distribution.
    """
    if not numbers:
        return 0.0
    s = sorted(numbers)
    # Correct:  idx = (p / 100) * (len(s) - 1)
    idx = (p / 100) * len(s)                                     # <── BUG 3
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


# ── 4. Z-score ─────────────────────────────────────────────────────────────

def z_score(value: float, numbers: list) -> float:
    """
    Standardised z-score.
    BUG #4: no guard against zero standard deviation → ZeroDivisionError
    when all values are identical.
    """
    m = mean(numbers)
    sd = std_dev(numbers)
    return (value - m) / sd                                       # <── BUG 4


# ── 5. Moving average ──────────────────────────────────────────────────────

def moving_average(numbers: list, window: int) -> list:
    """
    Simple moving average.
    BUG #5: slice is numbers[i-window : i] – off by one, should be
    numbers[i-window+1 : i+1]  (misses the current element).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    result = []
    for i in range(len(numbers)):
        if i < window:
            result.append(mean(numbers[: i + 1]))
        else:
            result.append(mean(numbers[i - window : i]))         # <── BUG 5
    return result


# ── 6. Correct helpers (no bugs) ──────────────────────────────────────────

def mode(numbers: list):
    """Most common value (first if tie)."""
    if not numbers:
        return None
    counts: dict = {}
    for n in numbers:
        counts[n] = counts.get(n, 0) + 1
    return max(counts, key=counts.get)


def normalize(numbers: list) -> list:
    """Min-max normalise to [0, 1]."""
    lo, hi = min(numbers), max(numbers)
    if hi == lo:
        return [0.0] * len(numbers)
    return [(x - lo) / (hi - lo) for x in numbers]


def covariance(xs: list, ys: list) -> float:
    """Sample covariance of two equal-length sequences."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Sequences must be equal length >= 2")
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
