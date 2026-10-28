class ParkingCounter:
    @staticmethod
    def count_slots(results, slots_config=None):
        enabled_ids = None

        if slots_config is not None:
            enabled_ids = {
                slot["id"]
                for slot in slots_config
                if slot.get("enabled", True)
            }

        counts = {
            "configured": 0,
            "valid": 0,
            "empty": 0,
            "occupied": 0,
            "unknown": 0,
            "invalid": 0,
        }

        for item in results:
            slot_id = item.get(
                "index",
                item.get("id"),
            )

            if (
                enabled_ids is not None
                and slot_id not in enabled_ids
            ):
                continue

            counts["configured"] += 1

            state = str(
                item.get(
                    "state",
                    item.get("label", "unknown"),
                )
            ).lower()

            if state == "empty":
                counts["empty"] += 1
                counts["valid"] += 1

            elif state in {"occupied", "not_empty", "car"}:
                counts["occupied"] += 1
                counts["valid"] += 1

            elif state == "invalid":
                counts["invalid"] += 1

            else:
                counts["unknown"] += 1
                counts["valid"] += 1

        return counts