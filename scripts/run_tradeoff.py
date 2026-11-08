"""Sec. 5 on one dataset: Fig. 3 trade-off, Table 2 row, Fig. 2 case study, Fig. 4 baseline.

Needs ``data/tables/<DS>_{train,test}.npz`` (prepare_data.py) and ``data/scores/<DS>_*.npz``
(train_scorers.py). Writes to ``outputs/<DS>/``. No API calls.

    uv run python scripts/run_tradeoff.py --dataset HEADLINES
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from frugalgpt.baselines import best_single_llm, cheapest_at_accuracy, threshold_curve
from frugalgpt.cascade import Cascade, simulate
from frugalgpt.data import SHORT_NAMES, load_tables
from frugalgpt.experiment import (
    Split,
    budget_grid,
    case_study,
    cost_to_match,
    match_support,
    robust_cost_to_match,
    tradeoff,
)
from frugalgpt.plots import plot_baseline, plot_case_study, plot_tradeoff
from frugalgpt.scorer import queries_hash

ROOT = Path(__file__).resolve().parents[1]
# Official budget grid (FrugalGPT_gen_tradeoff_HEADLINES*.ipynb). The FrugalGPT rows in
# summary/headlines_e8_perf_seed_9_2024updatedmodel.csv use the 2024 model pool, so they are not
# comparable to the 2023 pool here; we compare with the paper's 2023 numbers instead.
OFFICIAL_GRID = np.linspace(3.5e-5, 0.0036, 20)
N_LOG_OTHER = 30  # other datasets: log-spaced budgets from the cheapest to the best LLM's cost
LOW_FACTOR = 1.01  # grid starts this far above the cheapest LLM's cost (1.001 was never feasible)
ROBUST_LEVEL = 0.9  # Table 2 robust read-out: bootstrap support needed to count as a match
# Paper Table 2 (2023 pool): best single LLM, its test-set cost, FrugalGPT's cost at equal accuracy.
TABLE2_PAPER = {
    "HEADLINES": ("GPT-4", 33.1, 0.6),
    "OVERRULING": ("GPT-4", 9.7, 2.6),
    "AGNEWS": ("GPT-4", 64.6, 15.9),
    "COQA": ("GPT-3", 72.5, 29.6),
    "SCIQ": ("GPT-3", 132.4, 63.1),
}
FIG2_PAPER = {  # HEADLINES case study
    "chain": ("GPT-J", "J1-L", "GPT-4"),
    "cost_saving": 0.80,
    "acc_gain_pts": 1.5,
    "reach_gpt4": 0.166,
}


def load_split(ds: str, split: str):
    """Data tables + combined scores for one split, checked to describe the same rows."""
    t = load_tables(ROOT / "data" / "tables" / f"{ds}_{split}.npz")
    s = np.load(ROOT / "data" / "scores" / f"{ds}_{split}.npz")
    if tuple(s["llms"]) != t.llms:
        raise SystemExit("score columns do not match the data tables; rerun train_scorers.py")
    if "queries_hash" not in s or str(s["queries_hash"]) != queries_hash(t.queries):
        raise SystemExit("score rows do not match the data tables; rerun train_scorers.py")
    if np.isnan(s["score"]).any():
        raise SystemExit("some LLMs have no scores yet; run train_scorers.py")
    return t, Split(t.correct, t.cost, s["score"])


def _case_dict(cs, cs_pruned, n_test: int) -> dict:
    """Fig. 2 numbers as JSON."""
    return {
        "budget_per_query": cs.budget,
        "budget_total": cs.budget * n_test,
        "chain": cs.chain,
        "thresholds": cs.cascade.thresholds,
        "test_acc": cs.test_acc,
        "test_total_cost": cs.test_total_cost,
        "ref_llm": cs.ref_llm,
        "ref_acc": cs.ref_acc,
        "ref_total_cost": cs.ref_total_cost,
        "cost_saving": cs.cost_saving,
        "acc_gain_pts": cs.acc_gain * 100,
        "reach_share": cs.reach_share,
        "mpi_pruned_chain": None if cs_pruned is None else cs_pruned.chain,
        "mpi_pruned_test_acc": None if cs_pruned is None else cs_pruned.test_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="HEADLINES")
    ap.add_argument("--chain-search", default="exhaustive", choices=["exhaustive", "mpi_pruned"])
    ap.add_argument(
        "--n-log",
        type=int,
        default=None,
        help=f"log-spaced budgets (default: 10 on HEADLINES, {N_LOG_OTHER} elsewhere)",
    )
    args = ap.parse_args()
    ds = args.dataset
    out = ROOT / "outputs" / ds
    out.mkdir(parents=True, exist_ok=True)

    t_train, train = load_split(ds, "train")
    t_test, test = load_split(ds, "test")
    llms = [SHORT_NAMES.get(m, m) for m in t_train.llms]
    n_test = test.n

    single_acc = test.correct.mean(axis=0)
    single_total = test.cost.sum(axis=0)
    # Table 2 reference = most accurate LLM on **train** (no choice made by looking at test);
    # its test accuracy is the target. Same LLM as the paper on every dataset checked.
    ref = best_single_llm(train.correct, train.cost)
    ref_acc, ref_total = float(single_acc[ref]), float(single_total[ref])
    cheapest = int(np.argmin(train.cost.mean(axis=0)))
    # Fig. 2: 1/5 of the best LLM's cost. Taken from train so no test quantity sets the budget.
    fig2_budget = float(train.cost[:, ref].mean()) / 5

    # --- Fig. 3: trade-off curve ---------------------------------------------------------------
    # A budget equal to the cheapest LLM's own cost is never feasible (the quantile search always
    # passes the lowest-scored queries on), so the grid starts just above it.
    low = float(train.cost[:, cheapest].mean()) * LOW_FACTOR
    if ds == "HEADLINES":  # official HEADLINES grid + dense low end (step 9)
        n_log = 10 if args.n_log is None else args.n_log
        budgets = budget_grid(
            OFFICIAL_GRID[low < OFFICIAL_GRID], low, 4e-4, n_log, extra=[fig2_budget]
        )
    else:
        n_log = N_LOG_OTHER if args.n_log is None else args.n_log
        high = float(train.cost[:, ref].mean())
        budgets = budget_grid([], low, high, n_log, extra=[fig2_budget])
    t0 = time.time()
    curve = tradeoff(train, test, budgets, llms, chain_search=args.chain_search)
    print(f"trade-off: {len(budgets)} budgets in {time.time() - t0:.0f} s")
    curve.to_csv(out / "tradeoff.csv", index=False)

    plot_tradeoff(
        curve,
        single_acc,
        single_total,
        llms,
        out / "fig3.png",
        title=f"{ds}: accuracy vs. cost (Fig. 3)",
    )

    # --- Table 2 --------------------------------------------------------------------------------
    match = cost_to_match(curve, ref_acc)
    table2 = {
        "dataset": ds,
        "best_llm": llms[ref],
        "best_llm_acc": ref_acc,
        "best_llm_total_cost": ref_total,
        "frugal_total_cost": None if match is None else float(match["test_total_cost"]),
        "frugal_acc": None if match is None else float(match["test_acc"]),
        "frugal_chain": None if match is None else match["chain"],
        "saving": None if match is None else 1 - float(match["test_total_cost"]) / ref_total,
        "margin_queries": None
        if match is None
        else int(round((float(match["test_acc"]) - ref_acc) * n_test)),
    }
    # Margin-robust read-out: the point above is selected on noisy test accuracy (optimistic).
    robust = robust_cost_to_match(curve, test, ref, ROBUST_LEVEL)
    table2["robust"] = (
        {"level": ROBUST_LEVEL, "frugal_total_cost": None, "saving": None}
        if robust is None
        else {
            "level": ROBUST_LEVEL,
            "frugal_total_cost": float(robust.row["test_total_cost"]),
            "frugal_acc": float(robust.row["test_acc"]),
            "frugal_chain": robust.row["chain"],
            "saving": 1 - float(robust.row["test_total_cost"]) / ref_total,
            "margin_queries": robust.margin_queries,
            "support": robust.support,
        }
    )
    if match is not None:
        res = simulate(
            Cascade(tuple(match["order"]), tuple(match["thresholds"])),
            test.correct,
            test.cost,
            test.score,
        )
        table2["support"] = match_support(res.correct, test.correct[:, ref])
        # Sensitivity: drop test rows whose prompt is also in train (COQA: 21, from the release).
        seen = set(t_train.queries)
        fresh = np.array([q not in seen for q in t_test.queries])
        if not fresh.all():
            ref_fresh_cost = float(test.cost[fresh, ref].sum())
            table2["without_train_overlap"] = {
                "n_dropped": int((~fresh).sum()),
                "best_llm_acc": float(test.correct[fresh, ref].mean()),
                "frugal_acc": float(res.correct[fresh].mean()),
                "saving": 1 - float(res.cost[fresh].sum()) / ref_fresh_cost,
            }
    if ds in TABLE2_PAPER:
        p_ref, p_cost, p_frugal = TABLE2_PAPER[ds]
        table2["paper"] = {
            "best_llm": p_ref,
            "best_llm_total_cost": p_cost,
            "frugal_total_cost": p_frugal,
            "saving": 1 - p_frugal / p_cost,
        }
    (out / "table2.json").write_text(json.dumps(table2, indent=2))

    # --- Fig. 2: case study ---------------------------------------------------------------------
    cs = case_study(train, test, fig2_budget, llms, ref, chain_search=args.chain_search)
    cs_pruned = case_study(train, test, fig2_budget, llms, ref, chain_search="mpi_pruned")
    if cs is None:
        print(f"Fig. 2: no feasible strategy at budget {fig2_budget:.3g} per query; skipped")
        case = {"budget_per_query": fig2_budget, "chain": None}
    else:
        plot_case_study(cs, llms, out / "fig2.png")
        case = _case_dict(cs, cs_pruned, n_test)
    (out / "case_study.json").write_text(json.dumps(case, indent=2))

    # --- Fig. 4: simple threshold baseline (cheapest LLM's scorer score as stand-in) -------------
    s = test.score[:, cheapest]
    taus = np.unique(np.concatenate([[-1.0], np.quantile(s, np.linspace(0, 1, 201)), [1.0]]))
    base = threshold_curve(s, test.correct, test.cost, cheapest, ref, taus)
    base_match = cheapest_at_accuracy(base.accuracy, base.total_cost, ref_acc)
    base_label = f"{llms[cheapest]} → {llms[ref]}, one threshold on {llms[cheapest]}'s scorer"
    plot_baseline(
        curve,
        base.accuracy,
        base.total_cost,
        single_acc,
        single_total,
        llms,
        out / "fig4.png",
        base_label,
        title=f"{ds}: FrugalGPT vs. threshold baseline",
    )
    fig4 = {
        "baseline": base_label,
        "baseline_total_cost_to_match": None if base_match is None else base_match[1],
        "baseline_saving": None if base_match is None else 1 - base_match[1] / ref_total,
        "frugal_saving": table2["saving"],
    }
    (out / "fig4.json").write_text(json.dumps(fig4, indent=2))

    # --- report ---------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\nFig. 3 trade-off (test; cost = total over the test set, USD)")
    cols = ["budget", "chain", "train_acc", "test_acc", "test_total_cost"]
    print(curve[cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\nTable 2:", json.dumps(table2, indent=1))
    print("\nFig. 2:", json.dumps(case, indent=1, default=str))
    if ds == "HEADLINES":
        print("paper:", json.dumps(FIG2_PAPER))
    print("\nFig. 4:", json.dumps(fig4, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
