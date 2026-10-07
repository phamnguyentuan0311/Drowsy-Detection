"""
Tiền xử lý MRL Eye Dataset cho bài toán phân loại mắt Open/Closed.

Cấu trúc dataset gốc sau khi giải nén mrlEyes_2018_01.zip:
    data/raw/mrlEyes_2018_01/s0001/s0001_00001_0_0_0_0_0_01.png
    data/raw/mrlEyes_2018_01/s0002/...
    ...

Quy tắc đặt tên file:
    s<subjectID>_<imageID>_<gender>_<glasses>_<eyestate>_<reflections>_<lighting>_<sensorID>.png
    - eyestate: 0 = closed, 1 = open

Cách dùng:
    python prepare_dataset.py --raw_dir data/raw/mrlEyes_2018_01 --out_dir data/processed --img_size 64
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm

LABEL_MAP = {"0": "closed", "1": "open"}


def parse_label_from_filename(filename: str):
    """Trích xuất subject_id và eye_state (open/closed) từ tên file MRL."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 5:
        return None, None
    subject_id = parts[0]  # ví dụ: s0001
    eyestate_code = parts[4]
    eye_state = LABEL_MAP.get(eyestate_code)
    return subject_id, eye_state


def collect_images(raw_dir: Path):
    """Duyệt toàn bộ ảnh, gom theo subject_id -> list[(path, label)]."""
    by_subject = defaultdict(list)
    all_images = list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.jpg"))
    if not all_images:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh .png/.jpg trong {raw_dir}. "
            "Kiểm tra lại đường dẫn --raw_dir sau khi giải nén dataset."
        )
    for img_path in all_images:
        subject_id, label = parse_label_from_filename(img_path.name)
        if subject_id is None or label is None:
            continue
        by_subject[subject_id].append((img_path, label))
    return by_subject


def split_subjects(subject_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Chia danh sách subject_id thành train/val/test — tách theo NGƯỜI,
    không tách theo ảnh, để tránh rò rỉ dữ liệu giữa các tập."""
    subject_ids = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subjects = set(subject_ids[:n_train])
    val_subjects = set(subject_ids[n_train:n_train + n_val])
    test_subjects = set(subject_ids[n_train + n_val:])
    return train_subjects, val_subjects, test_subjects


def save_split(split_name, subject_ids, by_subject, out_dir: Path, img_size: int):
    stats = defaultdict(int)
    for subject_id in subject_ids:
        for img_path, label in by_subject[subject_id]:
            dst_dir = out_dir / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))

            dst_path = dst_dir / img_path.name
            cv2.imwrite(str(dst_path), img)
            stats[label] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="Tiền xử lý MRL Eye Dataset")
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Thư mục chứa dataset gốc đã giải nén (mrlEyes_2018_01)")
    parser.add_argument("--out_dir", type=str, default="data/processed",
                         help="Thư mục lưu dataset sau xử lý")
    parser.add_argument("--img_size", type=int, default=64,
                         help="Kích thước ảnh vuông sau resize (mặc định 64x64)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                         help="Xoá out_dir cũ trước khi tạo lại")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"Đang quét ảnh trong {raw_dir} ...")
    by_subject = collect_images(raw_dir)
    print(f"Tìm thấy {len(by_subject)} người (subject), "
          f"tổng {sum(len(v) for v in by_subject.values())} ảnh có nhãn hợp lệ.")

    train_subj, val_subj, test_subj = split_subjects(
        list(by_subject.keys()), args.train_ratio, args.val_ratio, args.seed
    )
    print(f"Chia theo người: train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)} người")

    for split_name, subj_set in [("train", train_subj), ("val", val_subj), ("test", test_subj)]:
        print(f"\nĐang xử lý tập {split_name} ...")
        stats = save_split(split_name, subj_set, by_subject, out_dir, args.img_size)
        print(f"  {split_name}: {dict(stats)}")

    print(f"\nHoàn tất. Dataset đã xử lý nằm ở: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
