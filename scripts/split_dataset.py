import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Đường dẫn thư mục
BASE_DIR = Path(".")
DATASET_DIR = BASE_DIR / "data" / "clf-data"
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "val"
TEST_DIR = BASE_DIR / "data" / "test"

# Tỉ lệ chia dataset
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

CLASSES = ["empty", "not_empty"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}

def get_group(path: Path) -> str:
    """Lấy phần trước dấu '_' làm mã nhóm (group ID) từ tên file."""
    return path.stem.split("_")[0]

def create_output_dirs():
    """Tạo thư mục đích nếu chưa tồn tại."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls_name in CLASSES:
            (split_dir / cls_name).mkdir(parents=True, exist_ok=True)

def main():
    print("=== BẮT ĐẦU CHIA DATASET THEO NHÓM VIDEO ===")
    create_output_dirs()

    # Gom nhóm các file theo class và theo group_id
    group_to_files = {cls: defaultdict(list) for cls in CLASSES}

    for cls_name in CLASSES:
        cls_dir = DATASET_DIR / cls_name
        if not cls_dir.exists():
            continue
        
        for file_path in cls_dir.iterdir():
            if file_path.suffix.lower() in VALID_EXTS:
                group_id = get_group(file_path)
                group_to_files[cls_name][group_id].append(file_path)

    # Chia dữ liệu đảm bảo không bị rò rỉ (data leakage) giữa các tập
    for cls_name in CLASSES:
        groups = list(group_to_files[cls_name].keys())
        random.seed(42)  # Cố định seed để kết quả đồng nhất khi chạy lại
        random.shuffle(groups)

        total_groups = len(groups)
        train_end = int(total_groups * TRAIN_RATIO)
        val_end = train_end + int(total_groups * VAL_RATIO)

        train_groups = groups[:train_end]
        val_groups = groups[train_end:val_end]
        test_groups = groups[val_end:]

        splits = [
            ("train", train_groups, TRAIN_DIR),
            ("val", val_groups, VAL_DIR),
            ("test", test_groups, TEST_DIR)
        ]

        print(f"\n--- Class: [{cls_name}] (Tổng số nhóm video: {total_groups}) ---")

        for split_name, group_list, target_base_dir in splits:
            count_files = 0
            target_dir = target_base_dir / cls_name

            for grp in group_list:
                files = group_to_files[cls_name][grp]
                for file_path in files:
                    dest_path = target_dir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    count_files += 1

            print(f" -> {split_name:<5}: {len(group_list):<3} nhóm | {count_files:<5} ảnh")

    print("\n=== HOÀN THÀNH CHIA DATASET THÀNH CÔNG ===")

if __name__ == "__main__":
    main()