import json

from web.insights import build_dashboard_insights


def write_model_report(reports_dir, occupied_support):
    model_dir = reports_dir / "metrics" / "cnn"
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.97,
                "precision": 0.91,
                "recall": 0.89,
                "f1_score": 0.90,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "classification_report.txt").write_text(
        "              precision    recall  f1-score   support\n"
        f"empty             0.98      0.96      0.97        25\n"
        f"occupied          0.90      0.90      0.90        {occupied_support}\n",
        encoding="utf-8",
    )


def write_run_report(reports_dir):
    run_dir = reports_dir / "runs" / "2026-11-01_205009"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "source": r"C:\private\parking.mp4",
                "processed_frames": 262,
                "processed_inferences": 132,
                "confidence_threshold": 0.6,
                "last_counts": {
                    "configured": 4,
                    "valid": 4,
                    "empty": 1,
                    "occupied": 2,
                    "unknown": 1,
                    "invalid": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_insights_warn_when_evaluation_report_has_no_occupied_examples(tmp_path):
    reports_dir = tmp_path / "reports"
    write_model_report(reports_dir, occupied_support=0)

    insights = build_dashboard_insights(reports_dir)

    assert insights["model_evaluation"]["supports"] == {"empty": 25, "occupied": 0}
    assert any(
        notice["level"] == "warning" and "không có mẫu lớp ô có xe" in notice["message"]
        for notice in insights["notices"]
    )
    assert any("Chưa tìm thấy summary.json" in notice["message"] for notice in insights["notices"])


def test_insights_summarize_latest_run_without_exposing_source_path(tmp_path):
    reports_dir = tmp_path / "reports"
    write_model_report(reports_dir, occupied_support=20)
    write_run_report(reports_dir)

    insights = build_dashboard_insights(reports_dir)
    run = insights["latest_run"]
    notice_text = " ".join(notice["message"] for notice in insights["notices"])

    assert run["source"] == "parking.mp4"
    assert run["processed_frames"] == 262
    assert run["processed_inferences"] == 132
    assert run["counts"]["unknown"] == 1
    assert "C:\\private" not in notice_text
    assert any(notice["title"] == "Một số ô chưa đủ độ tin cậy" for notice in insights["notices"])


def test_insights_handle_missing_reports_with_actionable_notices(tmp_path):
    insights = build_dashboard_insights(tmp_path / "reports")

    assert insights["model_evaluation"] is None
    assert insights["latest_run"] is None
    assert len(insights["notices"]) == 2
    assert all(notice["level"] == "info" for notice in insights["notices"])
