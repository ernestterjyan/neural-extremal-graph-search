"""Predeclared paired block analysis, simultaneous intervals for two comparisons."""

from __future__ import annotations

import math
import statistics

from .evidence import EVALUATION_SIZES


def student_cdf(t: float, degrees: int) -> float:
    if degrees < 1:
        raise ValueError("positive degrees of freedom required")
    if t < 0:
        return 1 - student_cdf(-t, degrees)
    normalizer = math.exp(math.lgamma((degrees + 1) / 2) - math.lgamma(degrees / 2)) / math.sqrt(
        degrees * math.pi
    )

    def density(x):
        return normalizer * (1 + x * x / degrees) ** (-(degrees + 1) / 2)

    def integrate(a, b, fa, fm, fb, whole, tolerance, depth):
        mid = (a + b) / 2
        left_mid, right_mid = density((a + mid) / 2), density((mid + b) / 2)
        left = (mid - a) * (fa + 4 * left_mid + fm) / 6
        right = (b - mid) * (fm + 4 * right_mid + fb) / 6
        change = left + right - whole
        if depth <= 0 or abs(change) <= 15 * tolerance:
            return left + right + change / 15
        return integrate(a, mid, fa, left_mid, fm, left, tolerance / 2, depth - 1) + integrate(
            mid, b, fm, right_mid, fb, right, tolerance / 2, depth - 1
        )

    if t == 0:
        return 0.5
    fa, fm, fb = density(0), density(t / 2), density(t)
    integral = integrate(0, t, fa, fm, fb, t * (fa + 4 * fm + fb) / 6, 1e-11, 25)
    return min(1.0, 0.5 + integral)


def student_quantile(probability: float, degrees: int) -> float:
    if not 0.5 <= probability < 1:
        raise ValueError("upper-half quantile required")
    low, high = 0.0, 1.0
    while student_cdf(high, degrees) < probability:
        high *= 2
    for _ in range(55):
        middle = (low + high) / 2
        if student_cdf(middle, degrees) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def paired_interval(differences: list[float], *, comparisons: int = 2) -> dict:
    if len(differences) != 8:
        raise ValueError("eight independent training-seed blocks required")
    mean = statistics.mean(differences)
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    critical = student_quantile(1 - 0.05 / (2 * comparisons), len(differences) - 1)
    return {
        "mean": mean,
        "lower": mean - critical * sem,
        "upper": mean + critical * sem,
        "blocks": differences,
        "simultaneous_confidence": 0.95,
        "method": "paired Student t with Bonferroni correction across designated comparisons",
        "assumptions": "independent model-seed blocks; approximately normal paired block means",
    }


def analyze(records: list[dict], sizes: list[int], *, checkpoint: int = 60) -> dict:
    if any(r["status"] != "complete" for r in records):
        raise ValueError("failed searches require explicit resolution; cannot be silently excluded")
    cells = {(r["method"], r["block"], r["n"], r["replication"]): r for r in records}
    methods = {r["method"] for r in records}
    if methods != {"gnn", "mlp", "tabu", "adaptive"}:
        raise ValueError("complete GNN, MLP, tabu and adaptive method panel required")
    required = {(m, b, n, i) for m in methods for b in range(8) for n in sizes for i in range(10)}
    if len(cells) != len(records) or set(cells) != required:
        raise ValueError("analysis requires complete eight-block, ten-replication panel")
    values = {key: r["checkpoints"][str(checkpoint)]["edges"] for key, r in cells.items()}
    comparators = ["tabu", "adaptive"]
    comparisons = {}
    for comparator in comparators:
        blocks = [
            statistics.mean(
                values[("gnn", b, n, i)] - values[(comparator, b, n, i)]
                for n in sizes
                for i in range(10)
            )
            for b in range(8)
        ]
        by_size = {
            str(n): statistics.mean(
                values[("gnn", b, n, i)] - values[(comparator, b, n, i)]
                for b in range(8)
                for i in range(10)
            )
            for n in sizes
        }
        comparisons[comparator] = {
            **paired_interval(blocks, comparisons=len(comparators)),
            "mean_advantage_by_size": by_size,
        }
    both_improved = sum(
        all(comparisons[c]["mean_advantage_by_size"][str(n)] > 0 for c in comparators)
        for n in sizes
    )
    primary_panel = (
        set(sizes) == set(EVALUATION_SIZES)
        and checkpoint == 60
        and set(comparators) == {"tabu", "adaptive"}
    )
    primary_passed = (
        primary_panel
        and both_improved >= 6
        and all(comparisons[c]["mean"] >= 1 and comparisons[c]["lower"] > 0 for c in comparators)
    )
    method_means = {
        method: {
            "mean_edges": statistics.mean(
                values[(method, b, n, i)] for b in range(8) for n in sizes for i in range(10)
            ),
            "mean_edges_by_size": {
                str(n): statistics.mean(
                    values[(method, b, n, i)] for b in range(8) for i in range(10)
                )
                for n in sizes
            },
        }
        for method in sorted(methods)
    }
    mlp_comparisons = {
        comparator: {
            **paired_interval(
                [
                    statistics.mean(
                        values[("mlp", b, n, i)] - values[(comparator, b, n, i)]
                        for n in sizes
                        for i in range(10)
                    )
                    for b in range(8)
                ],
                comparisons=3,
            ),
            "role": "secondary control; not a replacement for the GNN primary result",
        }
        for comparator in ["gnn", "tabu", "adaptive"]
    }
    return {
        "checkpoint_seconds": checkpoint,
        "sizes": sizes,
        "comparisons": comparisons,
        "method_means": method_means,
        "secondary_mlp_comparisons": mlp_comparisons,
        "sizes_better_than_all_comparators": both_improved,
        "primary_panel": primary_panel,
        "primary_passed": primary_passed,
        "learning_gate_passed": checkpoint == 60 and comparisons["adaptive"]["mean"] > 0,
    }
