"""Sắp xếp các ô đỗ thành cột (block) và dãy (lane) theo vị trí trên ảnh camera.

Quy tắc:
- Các ô có tâm x cách nhau nhiều hơn BLOCK_GAP px thì thuộc hai cột khác nhau.
- Trong mỗi cột, chia thành 2 dãy tại khoảng hở x lớn nhất.
- Trong mỗi dãy, sắp xếp từ trên xuống theo y.
"""

BLOCK_GAP = 15.0


def _center_x(slot):
    return slot["x"] + slot["width"] / 2


def _split_by_gap(items, key, gap):
    if not items:
        return []
    items = sorted(items, key=key)
    groups = [[items[0]]]
    for prev, cur in zip(items, items[1:]):
        if key(cur) - key(prev) > gap:
            groups.append([])
        groups[-1].append(cur)
    return groups


def _split_two_lanes(block):
    items = sorted(block, key=_center_x)
    if len(items) < 2:
        return [items]
    best_index, best_gap = None, 0.0
    for i in range(len(items) - 1):
        gap = _center_x(items[i + 1]) - _center_x(items[i])
        if gap > best_gap:
            best_index, best_gap = i, gap
    if best_gap <= 0:
        return [items]
    return [items[: best_index + 1], items[best_index + 1 :]]


def build_layout(slots):
    """Trả về danh sách cột, mỗi cột gồm hai dãy là danh sách id ô từ trên xuống."""
    enabled = [s for s in slots if s.get("enabled", True)]
    columns = []
    for block in _split_by_gap(enabled, _center_x, BLOCK_GAP):
        lanes = _split_two_lanes(block)
        lanes = [sorted(lane, key=lambda s: s["y"]) for lane in lanes]
        columns.append(lanes)

    return [
        {
            "column": index + 1,
            "lanes": [[slot["id"] for slot in lane] for lane in lanes],
        }
        for index, lanes in enumerate(columns)
    ]
