"""
Đánh giá model trên tập test: Accuracy, Precision, Recall, F1, Confusion Matrix.

Cách dùng:
    python evaluate_model.py --model_path models/eye_state_model.keras --data_dir data/processed
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt

IMG_SIZE = 96


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model trên tập test")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    print(f"Đang nạp model từ {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        labels="inferred",
        label_mode="binary",
        class_names=["closed", "open"],
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        shuffle=False,
    )

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).astype(int).flatten()
    y_prob = model.predict(test_ds).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix (hàng=thực tế, cột=dự đoán, thứ tự [closed, open]):")
    print(cm)
    print("\nBáo cáo chi tiết theo từng lớp:")
    print(classification_report(y_true, y_pred, target_names=["closed", "open"]))

    # Vẽ và lưu confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["closed", "open"])
    ax.set_yticklabels(["closed", "open"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title("Confusion Matrix - Eye State")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "confusion_matrix.png"
    fig.savefig(fig_path)
    print(f"\nĐã lưu confusion matrix tại: {fig_path}")


if __name__ == "__main__":
    main()
