"""
Thu thập ảnh mắt Open/Closed của chính bạn qua webcam để bổ sung vào dataset
(giúp model thích nghi với điều kiện webcam thật, giảm domain gap so với MRL).

Dùng MediaPipe FaceLandmarker để tự động crop vùng mắt trái/phải mỗi frame,
bạn chỉ cần bấm phím để gán nhãn cho frame hiện tại.

Cách dùng:
    python collect_custom_eye_data.py --out_dir data/custom

Phím tắt:
    'o' -> lưu frame hiện tại là "open" (mắt mở)
    'c' -> lưu frame hiện tại là "closed" (mắt nhắm)
    'q' -> thoát

Gợi ý khi thu thập (để dữ liệu đa dạng, model học tốt hơn):
    - Thu cả 2 mắt mở bình thường, mở to, nheo mắt, nhắm hờ, nhắm chặt
    - Đổi góc mặt (hơi nghiêng trái/phải/lên/xuống), đổi khoảng cách tới camera
    - Đổi ánh sáng nếu có thể (đèn phòng, gần cửa sổ, ánh sáng yếu)
    - Nên thu tối thiểu ~150-200 ảnh mỗi lớp (open/closed) để có tác dụng
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")

RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def ensure_face_landmarker_model(path: Path = FACE_LANDMARKER_MODEL_PATH) -> str:
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải model FaceLandmarker về {path} (chỉ cần tải 1 lần)...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
    print("Tải xong.")
    return str(path)


def get_landmark_points(landmarks, idx_list, img_w, img_h):
    return np.array([
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idx_list
    ])


def crop_eye(frame, landmarks, idx_list, img_w, img_h, margin=6):
    pts = get_landmark_points(landmarks, idx_list, img_w, img_h)
    x1, y1 = pts[:, 0].min() - margin, pts[:, 1].min() - margin
    x2, y2 = pts[:, 0].max() + margin, pts[:, 1].max() + margin
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_gray_crop(crop_bgr, out_dir: Path, label: str, counter: dict):
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    counter[label] += 1
    filename = label_dir / f"user_{int(time.time() * 1000)}_{counter[label]:05d}.png"
    cv2.imwrite(str(filename), gray)
    return True


def main():
    parser = argparse.ArgumentParser(description="Thu thập ảnh mắt open/closed của chính bạn")
    parser.add_argument("--out_dir", type=str, default="data/custom")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    counter = {"open": 0, "closed": 0}
    for label in ["open", "closed"]:
        existing = list((out_dir / label).glob("*.png")) if (out_dir / label).exists() else []
        counter[label] = len(existing)

    landmarker_model_path = ensure_face_landmarker_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Không mở được camera index {args.camera}.")
        return

    start_time = time.time()
    print("Đang chạy. Phím 'o' = lưu là MỞ, 'c' = lưu là NHẮM, 'q' = thoát.")
    print(f"Đã có sẵn: open={counter['open']}, closed={counter['closed']}")

    last_saved_msg = ""
    last_saved_time = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            display = frame.copy()
            right_crop = left_crop = None

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                right_crop = crop_eye(frame, landmarks, RIGHT_EYE_IDX, img_w, img_h)
                left_crop = crop_eye(frame, landmarks, LEFT_EYE_IDX, img_w, img_h)

                for idx_list in (RIGHT_EYE_IDX, LEFT_EYE_IDX):
                    pts = get_landmark_points(landmarks, idx_list, img_w, img_h)
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            else:
                cv2.putText(display, "Khong phat hien khuon mat", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(display, f"Da luu - Mo: {counter['open']}  Nham: {counter['closed']}",
                        (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display, "'o'=luu MO   'c'=luu NHAM   'q'=thoat",
                        (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if last_saved_msg and time.time() - last_saved_time < 0.8:
                cv2.putText(display, last_saved_msg, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Thu thap du lieu mat", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("o") or key == ord("c"):
                label = "open" if key == ord("o") else "closed"
                saved_any = False
                if save_gray_crop(right_crop, out_dir, label, counter):
                    saved_any = True
                if save_gray_crop(left_crop, out_dir, label, counter):
                    saved_any = True
                if saved_any:
                    last_saved_msg = f"Da luu: {label.upper()}"
                    last_saved_time = time.time()
                else:
                    last_saved_msg = "Khong crop duoc mat, thu lai"
                    last_saved_time = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print(f"\nTổng cộng đã thu thập: open={counter['open']}, closed={counter['closed']}")
        print(f"Lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
