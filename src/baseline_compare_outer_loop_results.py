"""Compare final reports from baseline outer-loop ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_final_report(path: Path) -> dict | None:
    report_path = path / "baseline_final_report.json"
    if not report_path.exists():
        print(f"[WARN] Missing {report_path}")
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline outer-loop strategies")
    parser.add_argument("--input_dirs", nargs="+", required=True, help="Output directories from baseline_outer_loop_ablation.py")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_outer_loop_comparison")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in args.input_dirs:
        report = load_final_report(Path(d))
        if report is not None:
            rows.append(report)

    if not rows:
        raise SystemExit("No baseline_final_report.json files found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["final_oracle_asr", "final_clean_macro_f1"], ascending=[True, False])
    df.to_csv(out / "baseline_outer_loop_final_comparison.csv", index=False)

    print("\nFINAL OUTER-LOOP COMPARISON")
    print(df.to_string(index=False))
    print(f"\n[INFO] Wrote {out / 'baseline_outer_loop_final_comparison.csv'}")

    plt.figure(figsize=(9, 5))
    plot_df = df.sort_values("final_oracle_asr", ascending=False)
    plt.barh(plot_df["strategy"], plot_df["final_oracle_asr"])
    plt.xlabel("Final Oracle ASR after 5 rounds (lower is better)")
    plt.ylabel("Training attacker strategy")
    plt.title("Final robustness by adversarial-training strategy")
    plt.tight_layout()
    plt.savefig(out / "baseline_outer_loop_final_oracle_asr.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["strategy"], plot_df["final_clean_macro_f1"])
    plt.xlabel("Final clean macro-F1")
    plt.ylabel("Training attacker strategy")
    plt.title("Clean performance after 5 rounds")
    plt.tight_layout()
    plt.savefig(out / "baseline_outer_loop_clean_macro_f1.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
