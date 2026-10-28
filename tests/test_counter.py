import pytest
from src.parking_counter import ParkingCounter

def test_parking_counter_standard():
    results = [
        {"index": 0, "state": "empty"},
        {"index": 1, "state": "occupied"},
        {"index": 2, "state": "unknown"},
        {"index": 3, "state": "invalid"}, # Sẽ bị bỏ qua
    ]
    
    counts = ParkingCounter.count_slots(results)
    
    assert counts["configured"] == 4
    assert counts["valid"] == 3
    assert counts["empty"] == 1
    assert counts["occupied"] == 1
    assert counts["unknown"] == 1
    assert counts["invalid"] == 1

def test_parking_counter_with_disabled_slots():
    results = [
        {"index": 0, "state": "empty"},
        {"index": 1, "state": "occupied"},
    ]
    # Slot số 1 bị disabled
    slots_config = [
        {"id": 0, "enabled": True},
        {"id": 1, "enabled": False}
    ]
    
    counts = ParkingCounter.count_slots(results, slots_config)
    
    assert counts["configured"] == 1
    assert counts["valid"] == 1
    assert counts["empty"] == 1
    assert counts["occupied"] == 0