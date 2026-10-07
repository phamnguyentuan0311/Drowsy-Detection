"""
Demo realtime CHỈ phát hiện ngáp bằng MAR (Mouth Aspect Ratio) — không cần
model, chỉ dùng landmark khuôn mặt (đúng yêu cầu đề tài: không train thêm
model cho phần ngáp).

MAR = khoảng cách dọc miệng / khoảng cách ngang miệng. Miệng càng há to (ngáp)
thì MAR càng lớn.

Cách dùng:
    python demo_yawn_realtime.py
Nhấn 'q' để thoát.
"""

import argparse
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ============================== CẤU HÌNH ==============================
MAR_THRESHOLD = 0.55             # MAR vượt ngưỡng này -> nghi ngờ đang há miệng/ngáp
                                  # QUAN TRỌNG: cần tự tinh chỉnh theo khuôn mặt/camera
                                  # thực tế của bạn — xem giá trị MAR hiển thị lúc nói
                                  # chuyện bình thường vs lúc ngáp thật để chọn ngưỡng
YAWN_MIN_FRAMES = 8              # số frame liên tiếp MAR vượt ngưỡng mới tính là 1 lần
                                  # ngáp (tránh đếm nhầm khi nói/cười/há miệng thoáng qua)
YAWN_WINDOW_SEC = 60.0           # cửa sổ thời gian đếm số lần ngáp gần đây

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")

# Landmark index (trong 478 điểm của FaceLandmarker) dùng để tính MAR.
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308

# ============================== TIỆN ÍCH ==============================


def ensure_face_landmarker_model(path: Path = FACE_LANDMARKER_MODEL_PATH) -> str:
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải model FaceLandmarker về {path} (chỉ cần tải 1 lần)...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
    print("Tải xong.")
    return str(path)


class AlertPlayer:
    """Phát âm thanh cảnh báo liên tục ở thread riêng, không chặn vòng lặp chính."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            if HAS_WINSOUND:
                winsound.Beep(700, 200)
            else:
                print("\a", end="", flush=True)
            time.sleep(0.3)


def compute_mar(landmarks, img_w, img_h):
    """Mouth Aspect Ratio = khoảng cách dọc / khoảng cách ngang của miệng."""
    top = np.array([landmarks[MOUTH_TOP].x * img_w, landmarks[MOUTH_TOP].y * img_h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * img_w, landmarks[MOUTH_BOTTOM].y * img_h])
    left = np.array([landmarks[MOUTH_LEFT].x * img_w, landmarks[MOUTH_LEFT].y * img_h])
    right = np.array([landmarks[MOUTH_RIGHT].x * img_w, landmarks[MOUTH_RIGHT].y * img_h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def get_mouth_points(landmarks, img_w, img_h):
    idx_list = [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]
    return [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idx_list]


class YawnCounter:
    """Đếm số lần ngáp trong cửa sổ thời gian gần đây, tránh đếm trùng 1 lần
    ngáp kéo dài thành nhiều lần (dùng hysteresis: phải quay về dưới ngưỡng
    rồi mới tính lần ngáp kế tiếp)."""

    def __init__(self):
        self.events = deque()
        self._streak = 0
        self._is_yawning = False

    def update(self, mar, now):
        just_completed = False
        if mar > MAR_THRESHOLD:
            self._streak += 1
            if self._streak >= YAWN_MIN_FRAMES and not self._is_yawning:
                self._is_yawning = True
                self.events.append(now)
                just_completed = True
        else:
            self._streak = 0
            self._is_yawning = False

        while self.events and now - self.events[0] > YAWN_WINDOW_SEC:
            self.events.popleft()

        return just_completed, self._is_yawning, len(self.events)


# ============================== MAIN ==============================


def main():
    parser = argparse.ArgumentParser(description="Demo realtime phát hiện ngáp bằng MAR")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

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

    alert = AlertPlayer()
    counter = YawnCounter()
    start_time = time.time()
    prev_time = time.time()
    print("Đang chạy — nhấn 'q' để thoát cửa sổ video.")
    print(f"Ngưỡng MAR hiện tại: {MAR_THRESHOLD} (chỉnh ở đầu file nếu cần)")

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

            now = time.time()
            mar = 0.0
            just_completed = False
            is_yawning = False
            yawn_count = 0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                mar = compute_mar(landmarks, img_w, img_h)
                just_completed, is_yawning, yawn_count = counter.update(mar, now)

                for x, y in get_mouth_points(landmarks, img_w, img_h):
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            else:
                cv2.putText(frame, "Khong phat hien khuon mat", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if just_completed:
                alert.start()
                # phát cảnh báo ngắn rồi tự dừng, không kéo dài liên tục như cảnh báo mắt
                threading.Timer(1.0, alert.stop).start()

            # ---- Vẽ overlay ----
            mar_color = (0, 0, 255) if is_yawning else (0, 200, 0)
            cv2.putText(frame, f"MAR: {mar:.3f}  (nguong: {MAR_THRESHOLD})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, mar_color, 2)
            cv2.putText(frame, f"Trang thai: {'DANG HA MIENG/NGAP' if is_yawning else 'Binh thuong'}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mar_color, 2)
            cv2.putText(frame, f"So lan ngap ({int(YAWN_WINDOW_SEC)}s gan day): {yawn_count}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # thanh MAR trực quan để dễ canh ngưỡng
            bar_x, bar_y, bar_w, bar_h = 20, 130, 300, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            fill_w = int(min(mar / 1.0, 1.0) * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), mar_color, -1)
            threshold_x = bar_x + int(min(MAR_THRESHOLD / 1.0, 1.0) * bar_w)
            cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_h + 5), (0, 165, 255), 2)

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if just_completed:
                cv2.putText(frame, "PHAT HIEN NGAP!", (20, img_h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Demo - Yawn Detection (MAR)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        alert.stop()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
