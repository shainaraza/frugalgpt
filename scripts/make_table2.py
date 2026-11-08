"""Combine ``outputs/<DS>/table2.json`` into ``outputs/table2.md`` (paper Table 2 vs. ours).

uv run python scripts/make_table2.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("HEADLINES", "OVERRULING", "COQA", "AGNEWS", "SCIQ")  # paper Table 2 order


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:.2f}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def table2_markdown(rows: list[dict]) -> str:
    """Table 2: best single LLM's test-set cost and FrugalGPT's cost at equal accuracy."""
    lines = [
        "| Dataset | Best LLM (ours / paper) | Its accuracy | Its cost (ours / paper) "
        "| FrugalGPT cost (ours / paper) | Saving (ours / paper) | FrugalGPT accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = r.get("paper", {})
        lines.append(
            f"| {r['dataset']} | {r['best_llm']} / {p.get('best_llm', '—')} "
            f"| {_pct(r['best_llm_acc'])} "
            f"| {_money(r['best_llm_total_cost'])} / {_money(p.get('best_llm_total_cost'))} "
            f"| {_money(r['frugal_total_cost'])} / {_money(p.get('frugal_total_cost'))} "
            f"| {_pct(r['saving'])} / {_pct(p.get('saving'))} "
            f"| {_pct(r['frugal_acc'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = []
    for ds in ORDER:
        path = ROOT / "outputs" / ds / "table2.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    md = table2_markdown(rows)
    (ROOT / "outputs" / "table2.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
