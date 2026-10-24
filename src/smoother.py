from collections import deque


class SlotSmoother:
    def __init__(self, window_size=7, min_stable_count=5):
        if window_size < 1:
            raise ValueError("window_size phải lớn hơn 0")
        if min_stable_count < 1 or min_stable_count > window_size:
            raise ValueError("min_stable_count phải nằm trong khoảng 1..window_size")

        self.window_size = window_size
        self.min_stable_count = min_stable_count
        self.histories = {}
        self.stable_states = {}

    def update(self, slot_id, current_state, confidence=None, min_confidence=0.5):
        if slot_id not in self.histories:
            self.histories[slot_id] = deque(maxlen=self.window_size)

        history = self.histories[slot_id]
        normalized_state = str(current_state).lower()
        if confidence is not None and confidence < min_confidence:
            return self.stable_states.get(slot_id, "unknown")

        history.append(normalized_state)
        last_state = history[-1]
        consecutive_count = 0
        for state in reversed(history):
            if state == last_state:
                consecutive_count += 1
            else:
                break

        if slot_id not in self.stable_states:
            self.stable_states[slot_id] = last_state
        elif consecutive_count >= self.min_stable_count:
            self.stable_states[slot_id] = last_state

        return self.stable_states[slot_id]

    def reset(self):
        self.histories.clear()
        self.stable_states.clear()