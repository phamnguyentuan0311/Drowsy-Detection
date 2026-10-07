"""
Gộp dữ liệu mắt tự thu thập (từ collect_custom_eye_data.py) vào tập TRAIN
của dataset đã xử lý. Chỉ thêm vào train, KHÔNG đụng vào val/test — để giữ
val/test làm chuẩn đánh giá khách quan, độc lập với dữ liệu mới thêm vào.

Cách dùng:
    python merge_custom_data.py --custom_dir data/custom --processed_dir data/processed
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Gộp dữ liệu mắt tự thu thập vào tập train")
    parser.add_argument("--custom_dir", type=str, default="data/custom")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    custom_dir = Path(args.custom_dir)
    processed_dir = Path(args.processed_dir)

    if not custom_dir.exists():
        print(f"Không tìm thấy {custom_dir}. Hãy chạy collect_custom_eye_data.py trước.")
        return

    total_copied = {"open": 0, "closed": 0}
    for label in ["open", "closed"]:
        src_dir = custom_dir / label
        if not src_dir.exists():
            continue
        dst_dir = processed_dir / "train" / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.glob("*.png"):
            shutil.copy2(img_path, dst_dir / img_path.name)
            total_copied[label] += 1

    print("Đã gộp vào tập train:")
    for label, n in total_copied.items():
        print(f"  {label}: +{n} ảnh")

    for split in ["train", "val", "test"]:
        for label in ["open", "closed"]:
            d = processed_dir / split / label
            n = len(list(d.glob("*.png"))) if d.exists() else 0
            print(f"{split}/{label}: {n} ảnh")


if __name__ == "__main__":
    main()
