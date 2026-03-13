from __future__ import annotations

from scripts.score_memorization import longest_contiguous_overlap


def test_longest_contiguous_overlap_counts_matching_runs() -> None:
    a = "this is a test of the overlap function".split()
    b = "we ran a test of the overlap function yesterday".split()
    assert longest_contiguous_overlap(a, b) == 6


def test_longest_contiguous_overlap_returns_zero_for_no_match() -> None:
    assert longest_contiguous_overlap(["alpha"], ["beta"]) == 0
