"""
Fine-tune tiếp model đã train (eye_state_model.keras) bằng dataset đã gộp
thêm ảnh mắt của chính bạn (qua collect_custom_eye_data.py + merge_custom_data.py).

Dùng learning rate rất nhỏ để tinh chỉnh nhẹ, tránh phá vỡ những gì model
đã học được từ tập MRL gốc — chỉ giúp model thích nghi thêm với đặc điểm
khuôn mặt/camera/ánh sáng thật của bạn.

Cách dùng:
    python finetune_custom.py --base_model_path models/eye_state_model.keras \
        --data_dir data/processed --out_dir models --epochs 5
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 96


def build_dataset(data_dir: Path, split: str, batch_size: int, shuffle: bool):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir / split,
        labels="inferred",
        label_mode="binary",
        class_names=["closed", "open"],
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )


def compute_weights(data_dir: Path):
    classes = []
    for label, name in [(0, "closed"), (1, "open")]:
        n = len(list((data_dir / "train" / name).glob("*.png")))
        classes.extend([label] * n)
    classes = np.array(classes)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=classes)
    return {0: weights[0], 1: weights[1]}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model bằng dữ liệu mắt tự thu thập")
    parser.add_argument("--base_model_path", type=str, required=True,
                         help="Model đã train trước đó (ví dụ models/eye_state_model.keras)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="Learning rate rất nhỏ để tinh chỉnh nhẹ, không phá model gốc")
    args = parser.parse_args()

    base_model_path = Path(args.base_model_path)
    if not base_model_path.exists():
        print(f"Không tìm thấy {base_model_path}.")
        print("Bạn cần có model đã train trước (chạy train_model.py trước nếu chưa có).")
        return

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Đang nạp model gốc từ {base_model_path} ...")
    model = tf.keras.models.load_model(base_model_path)

    print("Đang nạp dữ liệu (đã gộp thêm ảnh của bạn)...")
    train_ds = build_dataset(data_dir, "train", args.batch_size, shuffle=True)
    val_ds = build_dataset(data_dir, "val", args.batch_size, shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    class_weight = compute_weights(data_dir)
    print(f"Class weight: {class_weight}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    ckpt_path = out_dir / "eye_state_model_finetuned.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ]

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, class_weight=class_weight, callbacks=callbacks,
    )

    print(f"\nHoàn tất. Model đã fine-tune: {ckpt_path}")
    print("Chạy evaluate_model.py với --model_path trỏ tới file này để kiểm tra lại kết quả.")


if __name__ == "__main__":
    main()
