"""Sec. 5 figures from experiment outputs (matplotlib, saved as PNG). Costs are test-set totals."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from frugalgpt.experiment import CaseStudy  # noqa: E402

FRUGAL = "#2a6fdb"
BASE = "#e07b39"
SINGLE = "#555555"
OFFICIAL = "#999999"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _single_points(ax, acc: np.ndarray, total_cost: np.ndarray, labels: Sequence[str]) -> None:
    ax.scatter(total_cost, acc, color=SINGLE, s=25, zorder=3, label="single LLM")
    for a, c, name in zip(acc, total_cost, labels, strict=True):
        ax.annotate(name, (c, a), fontsize=7, xytext=(3, 3), textcoords="offset points")


def plot_tradeoff(
    curve: pd.DataFrame,
    single_acc: np.ndarray,
    single_total_cost: np.ndarray,
    labels: Sequence[str],
    path: Path,
    official: pd.DataFrame | None = None,
    title: str = "",
) -> None:
    """Fig. 3: test accuracy vs. test-set total cost, FrugalGPT curve + single-LLM points."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    c = curve.dropna(subset=["test_acc"]).sort_values("test_total_cost")
    ax.plot(c["test_total_cost"], c["test_acc"], "-o", color=FRUGAL, ms=3, label="FrugalGPT (ours)")
    if official is not None:
        o = official.sort_values("test_total_cost")
        ax.plot(
            o["test_total_cost"],
            o["test_acc"],
            "--s",
            color=OFFICIAL,
            ms=3,
            label="FrugalGPT (official CSV)",
        )
    _single_points(ax, single_acc, single_total_cost, labels)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("total cost on the test set (USD)")
    ax.set_ylabel("test accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_baseline(
    curve: pd.DataFrame,
    base_acc: np.ndarray,
    base_total_cost: np.ndarray,
    single_acc: np.ndarray,
    single_total_cost: np.ndarray,
    labels: Sequence[str],
    path: Path,
    base_label: str,
    title: str = "",
) -> None:
    """Fig. 4: FrugalGPT vs. the simple two-LLM threshold baseline."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    c = curve.dropna(subset=["test_acc"]).sort_values("test_total_cost")
    ax.plot(c["test_total_cost"], c["test_acc"], "-o", color=FRUGAL, ms=3, label="FrugalGPT")
    order = np.argsort(base_total_cost)
    ax.plot(base_total_cost[order], base_acc[order], "-", color=BASE, label=base_label)
    _single_points(ax, single_acc, single_total_cost, labels)
    ax.set_xlabel("total cost on the test set (USD)")
    ax.set_ylabel("test accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_case_study(cs: CaseStudy, labels: Sequence[str], path: Path) -> None:
    """Fig. 2: (c) accuracy / cost vs. the reference LLM, (d) MPI heatmap, (e) reach shares."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.6, 1]})

    ax = axes[0]
    names = [cs.ref_llm, "FrugalGPT"]
    ax.bar(names, [cs.ref_total_cost, cs.test_total_cost], color=[SINGLE, FRUGAL])
    for i, (cost, acc) in enumerate(
        [(cs.ref_total_cost, cs.ref_acc), (cs.test_total_cost, cs.test_acc)]
    ):
        ax.text(i, cost, f"${cost:.2f}\nacc {acc:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(cs.ref_total_cost, cs.test_total_cost) * 1.25)
    ax.set_ylabel("total test cost (USD)")
    ax.set_title(f"(c) cost −{cs.cost_saving:.0%}, accuracy {cs.acc_gain * 100:+.1f} pts")

    ax = axes[1]
    im = ax.imshow(cs.mpi, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("LLM that is right")
    ax.set_ylabel("LLM that is wrong")
    ax.set_title("(d) MPI (test split)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    ax.bar(range(len(cs.chain)), cs.reach_share, color=FRUGAL)
    ax.set_xticks(range(len(cs.chain)), cs.chain, rotation=20, fontsize=8)
    for i, s in enumerate(cs.reach_share):
        ax.text(i, s, f"{s:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("(e) share of queries reaching each LLM")
    _save(fig, path)
