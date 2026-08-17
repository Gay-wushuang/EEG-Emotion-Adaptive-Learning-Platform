"""Create descriptive session-level comparison without window-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROBABILITIES = ("prob_happy", "prob_normal", "prob_sad")


def non_overlapping_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    selected = []
    last_end = float("-inf")
    for _, row in frame.sort_values("signal_end_unix").iterrows():
        if float(row["signal_start_unix"]) >= last_end:
            selected.append(row)
            last_end = float(row["signal_end_unix"])
    return pd.DataFrame(selected)


def summarize(path: Path, session: str) -> list[dict]:
    data = pd.read_csv(path)
    stable = data[(~data["transition_window"]) & (data["stage"] != "unassigned")]
    rows = []
    for stage, group in stable.groupby("stage"):
        trusted = group[group["quality_level"] == "trusted"]
        blocks = non_overlapping_blocks(group)
        record = {
            "session": session,
            "stage": stage,
            "window_count": int(len(group)),
            "trusted_count": int(len(trusted)),
            "warning_count": int((group["quality_level"] == "warning").sum()),
            "low_ood_count": int((group["quality_level"] == "low_ood").sum()),
            "trusted_fraction": float(len(trusted) / len(group)),
            "nonoverlap_30s_blocks": int(len(blocks)),
        }
        for column in PROBABILITIES:
            record[f"{column}_all_mean"] = float(group[column].mean())
            record[f"{column}_trusted_mean"] = (
                float(trusted[column].mean()) if len(trusted) else None
            )
            record[f"{column}_nonoverlap_mean"] = (
                float(blocks[column].mean()) if len(blocks) else None
            )
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session1", type=Path, required=True)
    parser.add_argument("--session2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.session1, "session1") + summarize(args.session2, "session2")
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "session_stage_comparison.csv", index=False, encoding="utf-8-sig")
    wide = result.pivot(index="stage", columns="session")
    deltas = []
    for stage in sorted(result["stage"].unique()):
        first = result[(result["session"] == "session1") & (result["stage"] == stage)].iloc[0]
        second = result[(result["session"] == "session2") & (result["stage"] == stage)].iloc[0]
        deltas.append(
            {
                "stage": stage,
                "trusted_fraction_session1": first["trusted_fraction"],
                "trusted_fraction_session2": second["trusted_fraction"],
                "delta_prob_positive_all": second["prob_happy_all_mean"] - first["prob_happy_all_mean"],
                "delta_prob_neutral_all": second["prob_normal_all_mean"] - first["prob_normal_all_mean"],
                "delta_prob_negative_all": second["prob_sad_all_mean"] - first["prob_sad_all_mean"],
                "emotion_comparison_allowed": bool(
                    first["trusted_fraction"] >= 0.5 and second["trusted_fraction"] >= 0.5
                ),
            }
        )
    payload = {
        "independent_unit": "experiment_session",
        "supporting_unit": "non-overlapping 30-second blocks",
        "window_level_significance_forbidden": True,
        "emotion_comparison_rule": "both sessions require at least 50% trusted windows",
        "stage_deltas": deltas,
    }
    (args.output_dir / "session_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
