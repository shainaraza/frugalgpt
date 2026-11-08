import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "make_table2", Path(__file__).parents[1] / "scripts" / "make_table2.py"
)
mt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mt)


def test_table2_markdown_row_and_missing_values() -> None:
    rows = [
        {
            "dataset": "HEADLINES",
            "best_llm": "GPT-4",
            "best_llm_acc": 0.8556,
            "best_llm_total_cost": 33.31,
            "frugal_total_cost": 0.52,
            "frugal_acc": 0.8666,
            "saving": 0.984,
            "paper": {
                "best_llm": "GPT-4",
                "best_llm_total_cost": 33.1,
                "frugal_total_cost": 0.6,
                "saving": 0.9819,
            },
        },
        {
            "dataset": "X",
            "best_llm": "A",
            "best_llm_acc": 0.5,
            "best_llm_total_cost": 1.0,
            "frugal_total_cost": None,
            "frugal_acc": None,
            "saving": None,
        },
    ]
    md = mt.table2_markdown(rows).splitlines()
    assert len(md) == 4
    assert "| HEADLINES | GPT-4 / GPT-4 | 85.6% | $33.31 / $33.10 | $0.52 / $0.60 |" in md[2]
    assert "98.4% / 98.2%" in md[2]
    assert md[3].count("—") == 7  # no paper numbers, no matching point
