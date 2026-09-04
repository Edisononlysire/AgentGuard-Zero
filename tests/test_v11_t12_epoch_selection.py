from scripts.select_v11_t12_epoch import _selection_key, _summary


def _rows(*, safe_t1: int, safe_t2: int) -> list[dict]:
    rows = []
    for task, safe_count in (("T1", safe_t1), ("T2", safe_t2)):
        for index in range(10):
            safe = float(index < safe_count)
            rows.append(
                {
                    "task_id": task,
                    "safe_success": safe,
                    "attack_mitigation": safe,
                    "attack_success": 1.0 - safe,
                    "intent_accuracy": 1.0,
                    "overresponse": 0.0,
                    "unauthorized_high_impact": 0.0,
                    "business_cost": 0.1,
                }
            )
    return rows


def test_macro_safe_success_is_primary_epoch_selector() -> None:
    lower = _summary(_rows(safe_t1=5, safe_t2=5))
    higher = _summary(_rows(safe_t1=7, safe_t2=5))
    assert higher["t12_macro_safe_success"] > lower["t12_macro_safe_success"]
    assert _selection_key(higher) < _selection_key(lower)


def test_summary_keeps_t1_and_t2_separate() -> None:
    summary = _summary(_rows(safe_t1=8, safe_t2=2))
    assert summary["scenario_count_by_task"] == {"T1": 10, "T2": 10}
    assert summary["by_task"]["T1"]["safe_success"] == 0.8
    assert summary["by_task"]["T2"]["safe_success"] == 0.2
    assert summary["t12_macro_safe_success"] == 0.5
