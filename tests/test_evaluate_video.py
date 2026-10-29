import pandas as pd

from scripts.evaluate_video import evaluate_video


def test_evaluate_video_uses_stable_state(tmp_path):
    ground_truth = tmp_path / "ground_truth.csv"
    predictions = tmp_path / "slot_predictions.csv"

    pd.DataFrame(
        [
            {"frame": 1, "slot_id": 0, "label": "empty"},
            {"frame": 2, "slot_id": 0, "label": "not_empty"},
        ]
    ).to_csv(ground_truth, index=False)
    pd.DataFrame(
        [
            {
                "frame": 1,
                "slot_id": 0,
                "raw_state": "unknown",
                "stable_state": "empty",
            },
            {
                "frame": 2,
                "slot_id": 0,
                "raw_state": "occupied",
                "stable_state": "occupied",
            },
        ]
    ).to_csv(predictions, index=False)

    assert evaluate_video(ground_truth, predictions) == {
        "total_evaluated_samples": 2,
        "slot_accuracy": 1.0,
        "false_empty_rate": 0.0,
        "false_occupied_rate": 0.0,
    }


def test_evaluate_video_accepts_legacy_label_column(tmp_path):
    ground_truth = tmp_path / "ground_truth.csv"
    predictions = tmp_path / "slot_predictions.csv"

    pd.DataFrame([{"frame": 1, "slot_id": 0, "label": "occupied"}]).to_csv(
        ground_truth, index=False
    )
    pd.DataFrame([{"frame": 1, "slot_id": 0, "label": "not_empty"}]).to_csv(
        predictions, index=False
    )

    assert evaluate_video(ground_truth, predictions)["slot_accuracy"] == 1.0
