from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_emotion.report.runs import scan_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs", help="outputs directory to scan")
    p.add_argument("--topk", type=int, default=20, help="how many runs to keep in the summary")
    p.add_argument("--out", default="outputs/_summary", help="directory to write summary artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    records = scan_runs(args.outputs)
    if not records:
        raise SystemExit(f"No runs found in: {args.outputs}")

    # Build table
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "timestamp": r.timestamp,
                "model": r.model_type,
                "accuracy": r.accuracy,
                "run_dir": r.run_dir,
                "config_path": r.config_path or "",
                "best_params": json.dumps(r.best_params, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows).sort_values(["accuracy"], ascending=False).head(args.topk)
    csv_path = os.path.join(args.out, "runs_topk.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Group summary
    by_model = df.groupby("model")["accuracy"].agg(["count", "mean", "max"]).reset_index()
    by_model_path = os.path.join(args.out, "by_model.csv")
    by_model.to_csv(by_model_path, index=False, encoding="utf-8-sig")

    # Plot: max accuracy per model (default matplotlib colors)
    max_by_model = by_model.set_index("model")["max"].sort_values(ascending=False)
    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    max_by_model.plot(kind="bar", ax=ax)
    ax.set_ylabel("max accuracy")
    ax.set_title("Best accuracy by model")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "best_accuracy_by_model.png"), dpi=180)
    plt.close(fig)

    # Save json
    out_json = {
        "n_runs_scanned": len(records),
        "topk": int(args.topk),
        "best_overall": df.iloc[0].to_dict(),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote: {csv_path}")
    print(f"✅ Wrote: {by_model_path}")
    print(f"✅ Wrote: {os.path.join(args.out, 'best_accuracy_by_model.png')}")
    print(f"✅ Wrote: {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
