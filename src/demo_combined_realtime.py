"""
Demo realtime GỘP cả 2 phần: phát hiện mắt Open/Closed (model CNN, cảnh báo
khi nhắm mắt liên tục quá lâu) VÀ phát hiện ngáp (MAR, không cần model).

Dùng chung 1 lần detect landmark mỗi frame cho cả 2 phần, hiệu quả hơn chạy
2 demo riêng lẻ song song.

Cách dùng:
    python demo_combined_realtime.py --model_path models/eye_state_model_finetuned.keras
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
import tensorflow as tf
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

# ============================== CẤU HÌNH MẮT ==============================
EYE_MODEL_IMG_SIZE = 96
EYE_OPEN_PROB_THRESHOLD = 0.5
EYE_CLOSED_ALERT_SEC = 0.7        # nhắm mắt liên tục >= X giây -> cảnh báo
EYE_UNKNOWN_GRACE_SEC = 0.3        # nếu đang trong 1 đợt nhắm mắt mà thoáng bị
                                    # "không xác định" (rung lắc, 1 frame mờ...)
                                    # trong khoảng thời gian này, KHÔNG reset bộ
                                    # đếm — tránh mất đợt nhắm mắt chỉ vì chập
                                    # chờn 1-2 frame

MIN_EYE_BRIGHTNESS = 30
MAX_EYE_BRIGHTNESS = 235
# QUAN TRỌNG: từng thử dùng edge density (Canny) + contrast std để phát hiện
# vật cản che (tay, giấy...), nhưng cả 2 phép đo này đều có nguy cơ loại
# nhầm MẮT NHẮM THẬT (mắt nhắm cũng khá phẳng/ít tương phản, và viền mí/lông
# mi thật nhiều khi không đủ sắc nét để Canny bắt được ở ngưỡng cố định) —
# đã gặp lỗi thực tế do việc này. Vì ưu tiên hàng đầu là KHÔNG được bỏ sót
# mắt nhắm thật, nên bỏ hẳn kiểm tra edge density (không ổn định), chỉ giữ
# lại contrast std ở mức RẤT LỎNG — chỉ chặn được những vật cản cực kỳ
# phẳng mịn (giấy trắng trơn, bề mặt đồng màu tuyệt đối áp sát ống kính).
# Việc phát hiện vật cản vì vậy sẽ kém nhạy hơn — đây là đánh đổi có chủ
# đích, nên ghi rõ vào phần hạn chế của báo cáo.
MIN_CONTRAST_STD = 3.0

RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ============================== CẤU HÌNH NGÁP ==============================
MAR_THRESHOLD = 0.55              # cần tự tinh chỉnh theo khuôn mặt/camera thực tế
YAWN_MIN_FRAMES = 8               # số frame liên tiếp mới tính là 1 lần ngáp
YAWN_WINDOW_SEC = 60.0            # cửa sổ đếm số lần ngáp gần đây

MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308

# ============================== CẤU HÌNH ĐÁNH GIÁ BUỒN NGỦ TỔNG THỂ ========
# Kết hợp 3 tín hiệu trong 1 cửa sổ thời gian trượt để kết luận trạng thái
# buồn ngủ tổng thể — không kết luận từ 1 khung hình/1 lần nhắm mắt đơn lẻ.
TRACKING_WINDOW_SEC = 180.0       # cửa sổ thời gian trượt (3 phút, theo chuẩn PERCLOS
                                   # phổ biến trong các nghiên cứu phát hiện buồn ngủ)

PERCLOS_WARNING = 0.08            # % thời gian mắt nhắm trong cửa sổ -> WARNING
PERCLOS_DROWSY = 0.15             # % thời gian mắt nhắm trong cửa sổ -> DROWSY
                                   # (ngưỡng 15% là chuẩn PERCLOS được dùng phổ biến
                                   # nhất trong các nghiên cứu thực nghiệm lái xe)

LONG_CLOSURE_WARNING_COUNT = 2    # số lần nhắm mắt lâu (>=EYE_CLOSED_ALERT_SEC)
LONG_CLOSURE_DROWSY_COUNT = 4     # trong cửa sổ -> WARNING / DROWSY

YAWN_WARNING_COUNT_WINDOW = 2     # số lần ngáp trong cửa sổ -> WARNING
YAWN_DROWSY_COUNT_WINDOW = 3      # trong cửa sổ -> DROWSY

STATE_NORMAL, STATE_WARNING, STATE_DROWSY = "NORMAL", "WARNING", "DROWSY"
STATE_COLOR = {
    STATE_NORMAL: (0, 200, 0),
    STATE_WARNING: (0, 165, 255),
    STATE_DROWSY: (0, 0, 255),
}

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")

# ============================== TIỆN ÍCH CHUNG ==============================


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

    def __init__(self, freq=1000, duration_ms=300, interval_sec=0.4):
        self._stop_event = threading.Event()
        self._thread = None
        self.freq = freq
        self.duration_ms = duration_ms
        self.interval_sec = interval_sec

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
                winsound.Beep(self.freq, self.duration_ms)
            else:
                print("\a", end="", flush=True)
            time.sleep(self.interval_sec)


def get_landmark_points(landmarks, idx_list, img_w, img_h):
    return np.array([
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idx_list
    ])


# ------------------------------ Phần MẮT ------------------------------


def crop_eye(frame, landmarks, idx_list, img_w, img_h, margin=6):
    pts = get_landmark_points(landmarks, idx_list, img_w, img_h)
    x1, y1 = pts[:, 0].min() - margin, pts[:, 1].min() - margin
    x2, y2 = pts[:, 0].max() + margin, pts[:, 1].max() + margin
    # Vượt khung hình (bị vật cản/quay mặt che mất 1 phần) -> không dùng được,
    # không cắt xén cho vừa khung để tránh crop méo/thiếu gây dự đoán sai.
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def is_crop_usable(crop_bgr):
    """Kiểm tra crop mắt dùng được không: tồn tại, đủ sáng, không bị vật cản
    cực kỳ phẳng mịn che hoàn toàn (giấy trắng, bề mặt đồng màu tuyệt đối).

    Cố tình kiểm tra rất lỏng: ưu tiên không bao giờ loại nhầm mắt nhắm thật
    (xem ghi chú ở MIN_CONTRAST_STD phía trên)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if not (MIN_EYE_BRIGHTNESS <= mean_brightness <= MAX_EYE_BRIGHTNESS):
        return False
    contrast_std = float(np.std(gray))
    if contrast_std < MIN_CONTRAST_STD:
        return False
    return True


def predict_eye_state(model, eye_crop_bgr):
    """Model train trên ảnh GRAYSCALE -> phải convert grayscale rồi nhân 3
    kênh trước khi đưa vào model, khớp đúng phân phối dữ liệu lúc train."""
    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(gray_3ch, (EYE_MODEL_IMG_SIZE, EYE_MODEL_IMG_SIZE))
    debug_img = img.copy()
    img = np.expand_dims(img.astype(np.float32), axis=0)
    prob_open = float(model.predict(img, verbose=0)[0][0])
    return prob_open >= EYE_OPEN_PROB_THRESHOLD, prob_open, debug_img


def overlay_thumbnail(frame, thumb_rgb, x, y, size=90, label=""):
    if thumb_rgb is None:
        return
    thumb_bgr = cv2.cvtColor(cv2.resize(thumb_rgb, (size, size)), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    if x + size > w or y + size > h or x < 0 or y < 0:
        return
    frame[y:y + size, x:x + size] = thumb_bgr
    cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 1)
    if label:
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ------------------------------ Phần NGÁP ------------------------------


def compute_mar(landmarks, img_w, img_h):
    top = np.array([landmarks[MOUTH_TOP].x * img_w, landmarks[MOUTH_TOP].y * img_h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * img_w, landmarks[MOUTH_BOTTOM].y * img_h])
    left = np.array([landmarks[MOUTH_LEFT].x * img_w, landmarks[MOUTH_LEFT].y * img_h])
    right = np.array([landmarks[MOUTH_RIGHT].x * img_w, landmarks[MOUTH_RIGHT].y * img_h])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal if horizontal != 0 else 0.0


def get_mouth_points(landmarks, img_w, img_h):
    idx_list = [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]
    return [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idx_list]


class YawnCounter:
    """Đếm số lần ngáp trong cửa sổ thời gian gần đây, tránh đếm trùng 1 lần
    ngáp kéo dài thành nhiều lần."""

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


# ------------------------- Đánh giá buồn ngủ tổng thể -------------------------


class DrowsinessTracker:
    """Theo dõi trạng thái buồn ngủ tổng thể trong 1 cửa sổ thời gian trượt
    (TRACKING_WINDOW_SEC), kết hợp 3 tín hiệu:
      1. PERCLOS — % thời gian mắt nhắm trong cửa sổ
      2. Số lần nhắm mắt liên tục lâu (>=EYE_CLOSED_ALERT_SEC) trong cửa sổ
      3. Số lần ngáp trong cửa sổ

    Trạng thái DROWSY/WARNING được kết luận nếu BẤT KỲ tín hiệu nào chạm
    ngưỡng tương ứng — mỗi tín hiệu là một dấu hiệu đủ mạnh để cảnh báo độc
    lập, không cần cả 3 cùng lúc. Không kết luận từ 1 frame/1 sự kiện đơn lẻ.
    """

    def __init__(self):
        self.eye_samples = deque()    # (timestamp, is_closed: bool) — mỗi frame có mắt hợp lệ
        self.closure_events = deque()  # timestamp các lần "vừa đạt" ngưỡng nhắm lâu
        self.yawn_events = deque()     # timestamp các lần ngáp (lấy từ YawnCounter)
        self._was_alerting = False     # để chỉ đếm 1 lần khi vượt EYE_CLOSED_ALERT_SEC

    def update(self, is_open, is_eye_alert, now, yawn_just_completed):
        # 1. Ghi nhận mẫu mắt (chỉ khi xác định được, is_open is not None)
        if is_open is not None:
            self.eye_samples.append((now, not is_open))
        while self.eye_samples and now - self.eye_samples[0][0] > TRACKING_WINDOW_SEC:
            self.eye_samples.popleft()

        # 2. Ghi nhận 1 lần "nhắm mắt lâu" khi vừa chạm ngưỡng cảnh báo (cạnh lên)
        if is_eye_alert and not self._was_alerting:
            self.closure_events.append(now)
        self._was_alerting = is_eye_alert
        while self.closure_events and now - self.closure_events[0] > TRACKING_WINDOW_SEC:
            self.closure_events.popleft()

        # 3. Ghi nhận lần ngáp
        if yawn_just_completed:
            self.yawn_events.append(now)
        while self.yawn_events and now - self.yawn_events[0] > TRACKING_WINDOW_SEC:
            self.yawn_events.popleft()

        # ---- Tính các chỉ số ----
        perclos = (
            sum(1 for _, closed in self.eye_samples if closed) / len(self.eye_samples)
            if self.eye_samples else 0.0
        )
        closure_count = len(self.closure_events)
        yawn_count = len(self.yawn_events)

        # ---- Phân loại trạng thái ----
        if (perclos >= PERCLOS_DROWSY or closure_count >= LONG_CLOSURE_DROWSY_COUNT
                or yawn_count >= YAWN_DROWSY_COUNT_WINDOW):
            state = STATE_DROWSY
        elif (perclos >= PERCLOS_WARNING or closure_count >= LONG_CLOSURE_WARNING_COUNT
                or yawn_count >= YAWN_WARNING_COUNT_WINDOW):
            state = STATE_WARNING
        else:
            state = STATE_NORMAL

        return state, perclos, closure_count, yawn_count


# ============================== MAIN ==============================


def main():
    parser = argparse.ArgumentParser(description="Demo realtime: mắt + ngáp")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print("Đang nạp model phân loại mắt...")
    model = tf.keras.models.load_model(args.model_path)

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

    eye_alert = AlertPlayer(freq=1000, duration_ms=300, interval_sec=0.4)
    yawn_alert = AlertPlayer(freq=700, duration_ms=200, interval_sec=0.3)
    drowsy_alert = AlertPlayer(freq=1300, duration_ms=400, interval_sec=0.3)
    yawn_counter = YawnCounter()
    drowsiness_tracker = DrowsinessTracker()
    last_overall_state = STATE_NORMAL

    closed_since = None
    none_since = None
    prev_time = time.time()
    start_time = time.time()
    print("Đang chạy — nhấn 'q' để thoát cửa sổ video.")

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
            eye_label = "Khong nhan dien duoc khuon mat"
            right_prob = left_prob = None
            right_debug = left_debug = None
            is_open = None
            mar = 0.0
            yawn_just_completed = False
            is_yawning = False
            yawn_count = 0

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # ---- Mắt ----
                right_crop = crop_eye(frame, landmarks, RIGHT_EYE_IDX, img_w, img_h)
                left_crop = crop_eye(frame, landmarks, LEFT_EYE_IDX, img_w, img_h)

                if is_crop_usable(right_crop) and is_crop_usable(left_crop):
                    right_open, right_prob, right_debug = predict_eye_state(model, right_crop)
                    left_open, left_prob, left_debug = predict_eye_state(model, left_crop)
                    is_open = right_open and left_open
                    eye_label = "MO" if is_open else "NHAM"
                else:
                    eye_label = "Khong nhan dien duoc khuon mat"

                overlay_thumbnail(frame, right_debug, img_w - 200, 10, label="Phai")
                overlay_thumbnail(frame, left_debug, img_w - 100, 10, label="Trai")

                # ---- Ngáp ----
                mar = compute_mar(landmarks, img_w, img_h)
                yawn_just_completed, is_yawning, yawn_count = yawn_counter.update(mar, now)
                for x, y in get_mouth_points(landmarks, img_w, img_h):
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # ---- Cập nhật thời gian nhắm mắt liên tục ----
            if is_open is True:
                closed_since = None
                none_since = None
            elif is_open is False:
                if closed_since is None:
                    closed_since = now
                none_since = None
            else:  # is_open is None — không xác định được (chập chờn, vật cản...)
                if closed_since is not None:
                    # đang trong 1 đợt nhắm mắt -> cho khoan dung 1 khoảng
                    # ngắn trước khi coi là mất dấu thật sự
                    if none_since is None:
                        none_since = now
                    elif now - none_since > EYE_UNKNOWN_GRACE_SEC:
                        closed_since = None
                        none_since = None
            closed_duration = (now - closed_since) if closed_since else 0.0
            is_eye_alert = closed_duration >= EYE_CLOSED_ALERT_SEC

            if is_eye_alert:
                eye_alert.start()
            else:
                eye_alert.stop()

            if yawn_just_completed:
                yawn_alert.start()
                threading.Timer(1.0, yawn_alert.stop).start()

            overall_state, perclos, closure_count, window_yawn_count = drowsiness_tracker.update(
                is_open, is_eye_alert, now, yawn_just_completed
            )
            if overall_state != last_overall_state:
                print(f"[{time.strftime('%H:%M:%S')}] Trang thai tong the: "
                      f"{last_overall_state} -> {overall_state}  "
                      f"(PERCLOS={perclos*100:.1f}%, nham lau={closure_count} lan, "
                      f"ngap={window_yawn_count} lan trong {int(TRACKING_WINDOW_SEC)}s)")
                last_overall_state = overall_state

            if overall_state == STATE_DROWSY:
                drowsy_alert.start()
            else:
                drowsy_alert.stop()

            # ---- Vẽ overlay ----
            if is_eye_alert:
                eye_color = (0, 0, 255)
            elif is_open is None:
                eye_color = (128, 128, 128)
            elif is_open:
                eye_color = (0, 200, 0)
            else:
                eye_color = (0, 165, 255)

            cv2.putText(frame, f"Mat: {eye_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, eye_color, 2)
            if right_prob is not None:
                cv2.putText(frame, f"Xac suat mo - Phai: {right_prob:.2f}  Trai: {left_prob:.2f}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Nham lien tuc: {closed_duration:.1f}s", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            mar_color = (0, 0, 255) if is_yawning else (0, 200, 0)
            cv2.putText(frame, f"MAR: {mar:.2f}  So lan ngap ({int(YAWN_WINDOW_SEC)}s): {yawn_count}",
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)

            # ---- Trạng thái buồn ngủ tổng thể (dựa trên cửa sổ trượt) ----
            state_color = STATE_COLOR[overall_state]
            cv2.putText(frame, f"TRANG THAI TONG THE: {overall_state}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
            cv2.putText(frame,
                        f"PERCLOS: {perclos*100:.1f}%  Nham lau: {closure_count} lan  "
                        f"Ngap: {window_yawn_count} lan ({int(TRACKING_WINDOW_SEC/60)} phut gan day)",
                        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            border_color = None
            if is_eye_alert:
                border_color = (0, 0, 255)
                cv2.putText(frame, "CANH BAO: DANG NHAM MAT QUA LAU!", (20, img_h - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            if yawn_just_completed:
                cv2.putText(frame, "PHAT HIEN NGAP!", (20, img_h - 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                if border_color is None:
                    border_color = (0, 165, 255)

            if border_color:
                cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1), border_color, 8)

            cv2.imshow("Demo - Mat + Ngap", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        eye_alert.stop()
        yawn_alert.stop()
        drowsy_alert.stop()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
