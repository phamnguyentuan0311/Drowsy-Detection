"""
Train model phân loại mắt Open/Closed bằng Transfer Learning (MobileNetV2).

Input: dataset đã qua prepare_dataset.py, cấu trúc:
    data/processed/{train,val,test}/{open,closed}/*.png

Cách dùng:
    python train_model.py --data_dir data/processed --out_dir models --epochs 15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 96  # kích thước tối thiểu MobileNetV2 hỗ trợ trọng số ImageNet


def build_dataset(data_dir: Path, split: str, batch_size: int, shuffle: bool):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / split,
        labels="inferred",
        label_mode="binary",
        class_names=["closed", "open"],  # closed=0, open=1
        color_mode="rgb",  # MobileNetV2 cần ảnh 3 kênh
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )
    return ds


def build_model(fine_tune: bool = False, fine_tune_layers: int = 30):
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune
    if fine_tune:
        # chỉ mở khóa các layer cuối để fine-tune, giữ nguyên phần đầu
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compute_weights(data_dir: Path):
    """Tính class weight để bù lệch lớp (train có closed=31334, open=38538)."""
    classes = []
    for label, name in [(0, "closed"), (1, "open")]:
        n = len(list((data_dir / "train" / name).glob("*.png")))
        classes.extend([label] * n)
    classes = np.array(classes)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=classes)
    return {0: weights[0], 1: weights[1]}


def main():
    parser = argparse.ArgumentParser(description="Train model Open/Closed eye (MobileNetV2)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5,
                         help="Số epoch fine-tune thêm sau khi train phần đầu (0 = bỏ qua)")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Đang nạp dữ liệu...")
    train_ds = build_dataset(data_dir, "train", args.batch_size, shuffle=True)
    val_ds = build_dataset(data_dir, "val", args.batch_size, shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    class_weight = compute_weights(data_dir)
    print(f"Class weight (bù lệch lớp): {class_weight}")

    print("\n=== Giai đoạn 1: train phần đầu (base model đóng băng) ===")
    model, base_model = build_model(fine_tune=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    ckpt_path = out_dir / "eye_state_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, class_weight=class_weight, callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        print("\n=== Giai đoạn 2: fine-tune các layer cuối của MobileNetV2 ===")
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr / 10),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")],
        )
        history_ft = model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.fine_tune_epochs, class_weight=class_weight, callbacks=callbacks,
        )

    model.save(out_dir / "eye_state_model_final.keras")

    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(out_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nHoàn tất. Model tốt nhất: {ckpt_path}")
    print(f"Model cuối cùng (sau fine-tune): {out_dir / 'eye_state_model_final.keras'}")
    print("Chạy evaluate_model.py để đánh giá trên tập test.")


if __name__ == "__main__":
    main()
