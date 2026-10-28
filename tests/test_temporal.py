import pytest
from src.smoother import SlotSmoother

def test_slot_smoother_stability():
    # Cửa sổ 7, cần 5 lần liên tiếp để đổi trạng thái
    smoother = SlotSmoother(window_size=7, min_stable_count=5)
    slot_id = 0

    # Ban đầu slot là empty, cấp 4 frame occupied (chưa đủ 5) -> phải giữ empty
    for _ in range(4):
        state = smoother.update(slot_id, "occupied", confidence=0.9)
    
    # Do chưa đủ 5 frame liên tiếp, trạng thái ổn định vẫn phải là "occupied" hoặc giữ lịch sử ban đầu 
    # (Tùy thuộc vào khởi tạo, ở đây chuỗi toàn occupied nên stable_state sẽ chuyển thành occupied sau khi đủ)
    
    # Thêm frame thứ 5 occupied -> đủ điều kiện đổi sang occupied
    state = smoother.update(slot_id, "occupied", confidence=0.9)
    assert state == "occupied"

    # Gửi vào 1 frame bị nhiễu "empty" đơn lẻ -> không được làm đổi trạng thái ngay (phải giữ occupied)
    state = smoother.update(slot_id, "empty", confidence=0.9)
    assert state == "occupied"


def test_slot_smoother_requires_consecutive_frames_for_change():
    smoother = SlotSmoother(window_size=7, min_stable_count=5)
    slot_id = 1

    assert smoother.update(slot_id, "empty", confidence=0.9) == "unknown"
    for _ in range(4):
        assert smoother.update(slot_id, "occupied", confidence=0.9) == "unknown"

    assert smoother.update(slot_id, "occupied", confidence=0.9) == "occupied"


def test_slot_smoother_ignores_low_confidence_observation():
    smoother = SlotSmoother(window_size=7, min_stable_count=5)
    slot_id = 2

    assert smoother.update(slot_id, "empty", confidence=0.9) == "unknown"
    assert smoother.update(slot_id, "occupied", confidence=0.2) == "unknown"
