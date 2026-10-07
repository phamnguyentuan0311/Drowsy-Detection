"""
Test riêng model mắt Open/Closed, KHÔNG dùng MediaPipe — để loại trừ khả năng
lỗi do crop landmark bị lệch, cô lập xem model tự nó dự đoán đúng/sai thế nào.

Camera hiển thị ở chế độ ẢNH XÁM (đúng loại dữ liệu model đã được train).
Có 1 khung vuông cố định giữa màn hình — bạn tự di chuyển mặt/mắt để mắt nằm
gọn trong khung đó, model sẽ predict liên tục trên đúng vùng trong khung.

Cách dùng:
    python test_model_grayscale.py --model_path models/eye_state_model.keras
Phím tắt:
    'q' để thoát
    '+' / '-' để tăng/giảm kích thước khung
    phím mũi tên (arrow keys) để di chuyển khung
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf

EYE_MODEL_IMG_SIZE = 96


def predict(model, gray_crop):
    """Đưa ảnh xám (grayscale, đã crop) vào model, trả về (is_open, prob_open)."""
    gray_3ch = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(gray_3ch, (EYE_MODEL_IMG_SIZE, EYE_MODEL_IMG_SIZE))
    img = np.expand_dims(img.astype(np.float32), axis=0)
    prob_open = float(model.predict(img, verbose=0)[0][0])
    return prob_open >= 0.5, prob_open


def main():
    parser = argparse.ArgumentParser(description="Test model mắt trực tiếp bằng khung thủ công")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--box_size", type=int, default=160, help="Kích thước khung ban đầu (pixel)")
    args = parser.parse_args()

    print("Đang nạp model...")
    model = tf.keras.models.load_model(args.model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Không mở được camera index {args.camera}.")
        return

    box_size = args.box_size
    box_x, box_y = None, None  # sẽ đặt giữa khung hình ở frame đầu tiên

    print("Đang chạy — canh mắt vào khung vuông màu vàng.")
    print("Phím: 'q' thoát | '+'/'-' đổi kích thước khung | mũi tên di chuyển khung")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape

        if box_x is None:
            box_x, box_y = img_w // 2 - box_size // 2, img_h // 2 - box_size // 2

        box_x = max(0, min(box_x, img_w - box_size))
        box_y = max(0, min(box_y, img_h - box_size))

        crop = gray[box_y:box_y + box_size, box_x:box_x + box_size]

        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # để vẽ được chữ/khung màu

        label, prob = "Khong du du lieu", None
        if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
            is_open, prob = predict(model, crop)
            label = "MO" if is_open else "NHAM"

        color = (0, 255, 255)  # vàng
        cv2.rectangle(display, (box_x, box_y), (box_x + box_size, box_y + box_size), color, 2)

        text_color = (0, 200, 0) if label == "MO" else (0, 0, 255)
        cv2.putText(display, f"Du doan: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        if prob is not None:
            cv2.putText(display, f"Xac suat MO: {prob:.3f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(display, f"Khung: {box_size}px  (canh mat vao khung vang)",
                    (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # hiển thị thêm ảnh crop 96x96 thực tế đưa vào model, phóng to để nhìn rõ
        if crop.size > 0:
            debug_view = cv2.resize(crop, (150, 150))
            debug_view_bgr = cv2.cvtColor(debug_view, cv2.COLOR_GRAY2BGR)
            display[10:160, img_w - 160:img_w - 10] = debug_view_bgr
            cv2.rectangle(display, (img_w - 160, 10), (img_w - 10, 160), (255, 255, 255), 1)
            cv2.putText(display, "Input model", (img_w - 160, 5 + 170 - 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Test model - che do anh xam", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            box_size = min(box_size + 10, min(img_w, img_h))
        elif key == ord("-"):
            box_size = max(box_size - 10, 20)
        elif key == 81 or key == 2424832:   # mũi tên trái (khác biệt giữa các hệ điều hành)
            box_x -= 10
        elif key == 83 or key == 2555904:   # mũi tên phải
            box_x += 10
        elif key == 82 or key == 2490368:   # mũi tên lên
            box_y -= 10
        elif key == 84 or key == 2621440:   # mũi tên xuống
            box_y += 10

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
