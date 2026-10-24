class ParkingCounter:
    @staticmethod
    def count_slots(results, slots_config=None):
        """
        Thực hiện đếm số lượng trạng thái các ô đỗ xe từ kết quả phân loại.
        
        :param results: Danh sách kết quả phân loại từ detector/classifier (mỗi item chứa state/label).
        :param slots_config: Danh sách config các slot (để kiểm tra trạng thái enabled nếu cần).
        :return: Dict chứa tổng số lượng theo từng trạng thái.
        """
        empty_count = 0
        occupied_count = 0
        unknown_count = 0
        total_valid_count = 0

        # Tạo một map tra cứu trạng thái enabled từ config nếu có cung cấp
        enabled_map = {}
        if slots_config:
            for s in slots_config:
                enabled_map[s.get("id")] = s.get("enabled", True)

        for item in results:
            slot_id = item.get("index", item.get("id"))
            
            # Kiểm tra nếu slot bị disabled trong cấu hình thì bỏ qua không đếm
            if slot_id in enabled_map and not enabled_map[slot_id]:
                continue

            state = str(item.get("state", item.get("label", "unknown"))).lower()

            # Không tính slot 'invalid' vào số slot hợp lệ
            if state == "invalid":
                continue

            if state == "empty":
                total_valid_count += 1
                empty_count += 1
            elif state in ["occupied", "not_empty", "car"]:
                total_valid_count += 1
                occupied_count += 1
            elif state != "invalid":
                total_valid_count += 1
                unknown_count += 1

        return {
            "total": total_valid_count,
            "empty": empty_count,
            "occupied": occupied_count,
            "unknown": unknown_count,
        }