"""
Ứng dụng realtime phát hiện & cảnh báo buồn ngủ tài xế.

Pipeline:
    Webcam -> MediaPipe FaceMesh (landmark) -> crop mắt -> model CNN (open/closed)
                                             -> landmark miệng -> MAR (ngáp)
    -> Thuật toán buồn ngủ (kết hợp mắt + ngáp + thời gian) -> NORMAL/WARNING/DROWSY
    -> Cảnh báo (âm thanh + hiển thị) + lưu lịch sử vào CSV

Cách dùng:
    python realtime_app.py --model_path models/eye_state_model.keras
Nhấn 'q' để thoát.
"""

import argparse
import csv
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False  # không phải Windows -> cảnh báo chỉ hiển thị hình ảnh

# ============================== CẤU HÌNH ==============================
# Các ngưỡng dưới đây có thể chỉnh để phù hợp với từng người/điều kiện.
# Nên thử nghiệm và ghi lại giá trị đã chọn kèm lý do trong báo cáo.

EYE_MODEL_IMG_SIZE = 96          # phải khớp với IMG_SIZE lúc train model
EYE_OPEN_PROB_THRESHOLD = 0.5    # prob >= ngưỡng này -> coi là "open"

MAR_THRESHOLD = 0.55             # tỉ lệ mở miệng vượt ngưỡng này -> nghi ngờ đang ngáp
YAWN_MIN_FRAMES = 8              # số frame liên tiếp MAR vượt ngưỡng mới tính là 1 lần ngáp
                                  # (tránh đếm nhầm khi nói chuyện/cười thoáng qua)

EYE_WARNING_SEC = 1.0            # mắt nhắm liên tục >= X giây -> WARNING
EYE_DROWSY_SEC = 2.0             # mắt nhắm liên tục >= X giây -> DROWSY (vi giấc ngủ)

YAWN_WINDOW_SEC = 60.0           # cửa sổ thời gian để đếm số lần ngáp gần đây
YAWN_WARNING_COUNT = 2           # >= 2 lần ngáp trong cửa sổ -> WARNING
YAWN_DROWSY_COUNT = 3            # >= 3 lần ngáp trong cửa sổ -> DROWSY

STATE_NORMAL, STATE_WARNING, STATE_DROWSY = "NORMAL", "WARNING", "DROWSY"
STATE_COLOR = {
    STATE_NORMAL: (0, 200, 0),
    STATE_WARNING: (0, 165, 255),
    STATE_DROWSY: (0, 0, 255),
}

# Landmark index của MediaPipe FaceMesh (468 điểm) dùng để khoanh vùng mắt/miệng.
RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308

# ============================== TIỆN ÍCH ==============================


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
                winsound.Beep(1000, 300)  # tần số 1000Hz, 300ms
            else:
                print("\a", end="", flush=True)  # bell ký tự, dự phòng khi không có winsound
            time.sleep(0.4)


class HistoryLogger:
    """Ghi lịch sử sự kiện (đổi trạng thái, ngáp, nhắm mắt kéo dài) ra file CSV."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(["timestamp", "event_type", "detail"])
            self._file.flush()

    def log(self, event_type: str, detail: str = ""):
        self._writer.writerow([datetime.now().isoformat(timespec="seconds"), event_type, detail])
        self._file.flush()

    def close(self):
        self._file.close()


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


def predict_eye_state(model, eye_crop_bgr):
    """Trả về True nếu mắt được dự đoán là 'open'."""
    if eye_crop_bgr is None or eye_crop_bgr.size == 0:
        return None
    img = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (EYE_MODEL_IMG_SIZE, EYE_MODEL_IMG_SIZE))
    img = np.expand_dims(img.astype(np.float32), axis=0)
    prob_open = model.predict(img, verbose=0)[0][0]
    return prob_open >= EYE_OPEN_PROB_THRESHOLD


class DrowsinessMonitor:
    """Thuật toán buồn ngủ: kết hợp thời gian nhắm mắt liên tục + số lần ngáp gần đây."""

    def __init__(self, logger: HistoryLogger):
        self.logger = logger
        self.closed_since = None
        self.yawn_events = deque()  # timestamps các lần ngáp trong cửa sổ gần đây
        self._yawning_streak = 0    # đếm số frame liên tiếp MAR vượt ngưỡng
        self._is_yawning = False    # tránh đếm trùng 1 lần ngáp kéo dài thành nhiều lần
        self.state = STATE_NORMAL
        self._last_logged_state = STATE_NORMAL

    def update_eye(self, is_open, now):
        if is_open is None:
            return  # không phát hiện được mắt (quay mặt, ánh sáng...) -> bỏ qua frame này
        if is_open:
            if self.closed_since is not None:
                duration = now - self.closed_since
                if duration >= EYE_WARNING_SEC:
                    self.logger.log("eye_closure_episode", f"duration={duration:.2f}s")
            self.closed_since = None
        else:
            if self.closed_since is None:
                self.closed_since = now

    def update_mouth(self, mar, now):
        if mar > MAR_THRESHOLD:
            self._yawning_streak += 1
            if self._yawning_streak >= YAWN_MIN_FRAMES and not self._is_yawning:
                self._is_yawning = True
                self.yawn_events.append(now)
                self.logger.log("yawn_detected", f"MAR={mar:.2f}")
        else:
            self._yawning_streak = 0
            self._is_yawning = False

        # loại bỏ các lần ngáp đã ngoài cửa sổ thời gian quan sát
        while self.yawn_events and now - self.yawn_events[0] > YAWN_WINDOW_SEC:
            self.yawn_events.popleft()

    def eval_state(self, now):
        closed_duration = (now - self.closed_since) if self.closed_since else 0.0
        yawn_count = len(self.yawn_events)

        if closed_duration >= EYE_DROWSY_SEC or yawn_count >= YAWN_DROWSY_COUNT:
            new_state = STATE_DROWSY
        elif closed_duration >= EYE_WARNING_SEC or yawn_count >= YAWN_WARNING_COUNT:
            new_state = STATE_WARNING
        else:
            new_state = STATE_NORMAL

        if new_state != self._last_logged_state:
            self.logger.log("state_change", f"{self._last_logged_state} -> {new_state}")
            self._last_logged_state = new_state

        self.state = new_state
        return new_state, closed_duration, yawn_count


# ============================== MAIN ==============================


def main():
    parser = argparse.ArgumentParser(description="Realtime driver drowsiness detection")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--history_path", type=str, default="data/history_log.csv")
    args = parser.parse_args()

    print("Đang nạp model...")
    model = tf.keras.models.load_model(args.model_path)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Không mở được camera index {args.camera}.")
        return

    logger = HistoryLogger(Path(args.history_path))
    monitor = DrowsinessMonitor(logger)
    alert = AlertPlayer()

    prev_time = time.time()
    print("Đang chạy — nhấn 'q' để thoát cửa sổ video.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            now = time.time()
            eye_label = "Không phát hiện khuôn mặt"
            mar_value = 0.0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                right_crop = crop_eye(frame, landmarks, RIGHT_EYE_IDX, img_w, img_h)
                left_crop = crop_eye(frame, landmarks, LEFT_EYE_IDX, img_w, img_h)
                right_open = predict_eye_state(model, right_crop)
                left_open = predict_eye_state(model, left_crop)

                votes = [v for v in [right_open, left_open] if v is not None]
                is_open = (sum(votes) >= 1) if votes else None  # 1 trong 2 mắt "open" -> coi là open
                monitor.update_eye(is_open, now)
                eye_label = "Không xác định" if is_open is None else ("Mở" if is_open else "Nhắm")

                mar_value = compute_mar(landmarks, img_w, img_h)
                monitor.update_mouth(mar_value, now)

            state, closed_duration, yawn_count = monitor.eval_state(now)

            if state == STATE_DROWSY:
                alert.start()
            else:
                alert.stop()

            # ---- Vẽ overlay lên frame ----
            color = STATE_COLOR[state]
            cv2.putText(frame, f"Trang thai: {state}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Mat: {eye_label}  (nham lien tuc: {closed_duration:.1f}s)",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"MAR: {mar_value:.2f}  So lan ngap ({int(YAWN_WINDOW_SEC)}s gan day): {yawn_count}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if state == STATE_DROWSY:
                cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1), color, 8)

            cv2.imshow("Drowsy Driver Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        alert.stop()
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        logger.close()
        print(f"Đã lưu lịch sử tại: {args.history_path}")


if __name__ == "__main__":
    main()
