import argparse
import json
from pathlib import Path

import pandas as pd


def latest_prediction_csv(runs_dir):
    candidates = sorted(
        (path for path in runs_dir.glob("*/slot_predictions.csv") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No slot_predictions.csv found under {runs_dir}")
    return candidates[0]


def normalize_labels(labels):
    return (
        labels.astype(str)
        .str.strip()
        .str.lower()
        .replace({"not_empty": "occupied", "car": "occupied"})
    )


def evaluate_video(gt_path, pred_path):
    df_gt = pd.read_csv(gt_path)
    df_pred = pd.read_csv(pred_path)

    required_gt = {"frame", "slot_id", "label"}
    missing_gt = required_gt.difference(df_gt.columns)
    if missing_gt:
        raise ValueError(f"Ground truth CSV is missing columns: {sorted(missing_gt)}")

    pred_label_column = "stable_state" if "stable_state" in df_pred.columns else "label"
    required_pred = {"frame", "slot_id", pred_label_column}
    missing_pred = required_pred.difference(df_pred.columns)
    if missing_pred:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing_pred)}")

    key_columns = ["frame", "slot_id"]
    if df_gt.duplicated(key_columns).any():
        raise ValueError("Ground truth CSV has duplicate frame/slot_id rows")
    if df_pred.duplicated(key_columns).any():
        raise ValueError("Prediction CSV has duplicate frame/slot_id rows")

    ground_truth = df_gt[key_columns + ["label"]].rename(columns={"label": "label_gt"})
    predictions = df_pred[key_columns + [pred_label_column]].rename(
        columns={pred_label_column: "label_pred"}
    )
    merged = pd.merge(
        ground_truth,
        predictions,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No matching frame/slot_id rows between ground truth and predictions"
        )

    merged["label_gt"] = normalize_labels(merged["label_gt"])
    merged["label_pred"] = normalize_labels(merged["label_pred"])

    total_samples = len(merged)
    correct_predictions = (merged["label_gt"] == merged["label_pred"]).sum()
    total_occupied_gt = (merged["label_gt"] == "occupied").sum()
    total_empty_gt = (merged["label_gt"] == "empty").sum()
    false_empty = (
        (merged["label_gt"] == "occupied") & (merged["label_pred"] == "empty")
    ).sum()
    false_occupied = (
        (merged["label_gt"] == "empty") & (merged["label_pred"] == "occupied")
    ).sum()

    return {
        "total_evaluated_samples": int(total_samples),
        "slot_accuracy": round(float(correct_predictions / total_samples), 4),
        "false_empty_rate": round(
            float(false_empty / total_occupied_gt) if total_occupied_gt else 0.0,
            4,
        ),
        "false_occupied_rate": round(
            float(false_occupied / total_empty_gt) if total_empty_gt else 0.0,
            4,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate parking-slot predictions against video ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("reports/ground_truth.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Prediction CSV; defaults to the newest reports/runs/*/slot_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/metrics/video_evaluation_metrics.json"),
    )
    args = parser.parse_args()

    if not args.ground_truth.is_file():
        raise FileNotFoundError(f"Ground truth CSV not found: {args.ground_truth}")

    pred_path = args.predictions or latest_prediction_csv(Path("reports/runs"))
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")

    metrics_result = evaluate_video(args.ground_truth, pred_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(metrics_result, file, indent=4)

    print(f"Evaluated predictions: {pred_path}")
    print("=== VIDEO EVALUATION ===")
    for key, value in metrics_result.items():
        print(f" - {key}: {value}")
    print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
