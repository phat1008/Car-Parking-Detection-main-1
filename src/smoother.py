from collections import deque


class SlotSmoother:
    STABLE_STATES = {"empty", "occupied"}

    def __init__(
        self,
        window_size=7,
        min_stable_count=5,
    ):
        if window_size < 1:
            raise ValueError(
                "window_size phải lớn hơn 0"
            )

        if not 1 <= min_stable_count <= window_size:
            raise ValueError(
                "min_stable_count phải nằm trong 1..window_size"
            )

        self.window_size = window_size
        self.min_stable_count = min_stable_count
        self.histories = {}
        self.stable_states = {}

    def update(
        self,
        slot_id,
        current_state,
        confidence=None,
        min_confidence=0.5,
    ):
        state = str(current_state).lower()

        # Không đưa invalid/error/unknown vào lịch sử ổn định.
        if state not in self.STABLE_STATES:
            return self.stable_states.get(
                slot_id,
                "unknown",
            )

        if (
            confidence is not None
            and confidence < min_confidence
        ):
            return self.stable_states.get(
                slot_id,
                "unknown",
            )

        history = self.histories.setdefault(
            slot_id,
            deque(maxlen=self.window_size),
        )

        history.append(state)

        consecutive_count = 0

        for previous_state in reversed(history):
            if previous_state != state:
                break

            consecutive_count += 1

        if consecutive_count >= self.min_stable_count:
            self.stable_states[slot_id] = state

        return self.stable_states.get(
            slot_id,
            "unknown",
        )

    def current(self, slot_id):
        return self.stable_states.get(
            slot_id,
            "unknown",
        )

    def reset_slot(self, slot_id):
        self.histories.pop(slot_id, None)
        self.stable_states.pop(slot_id, None)

    def reset(self):
        self.histories.clear()
        self.stable_states.clear()