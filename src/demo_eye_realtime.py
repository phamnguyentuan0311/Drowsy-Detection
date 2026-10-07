"""
Demo realtime CHỈ phát hiện mắt Open/Closed bằng model đã train + cảnh báo.
(Không có phần phát hiện ngáp — dùng để test riêng model mắt.)

Dùng mediapipe FaceLandmarker (Tasks API) — API mới thay thế cho
`mediapipe.solutions.face_mesh` đã bị gỡ bỏ ở các bản mediapipe mới
(từ ~0.10.30 trở lên). Model landmark (.task) sẽ được tự động tải về
ở lần chạy đầu tiên (cần mạng).

Cách dùng:
    python demo_eye_realtime.py --model_path models/eye_state_model.keras
Nhấn 'q' để thoát.
"""

import argparse
import threading
import time
import urllib.request
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

# ============================== CẤU HÌNH ==============================
EYE_MODEL_IMG_SIZE = 96          # phải khớp với IMG_SIZE lúc train model
EYE_OPEN_PROB_THRESHOLD = 0.5    # prob >= ngưỡng này -> coi là "open"
EYE_CLOSED_ALERT_SEC = 0.7       # nhắm mắt liên tục >= X giây -> cảnh báo

MIN_EYE_BRIGHTNESS = 30          # độ sáng trung bình (0-255) tối thiểu để coi là đủ sáng
MAX_EYE_BRIGHTNESS = 235         # độ sáng trung bình tối đa (tránh cháy sáng/loá)
MIN_EDGE_DENSITY = 0.02          # tỉ lệ pixel biên tối thiểu — vùng bị vật cản che
                                  # thường rất phẳng mịn, gần như không có cạnh/chi tiết
MIN_CONTRAST_STD = 12.0          # độ lệch chuẩn độ sáng tối thiểu — mắt thật (kể cả nhắm)
                                  # luôn có vùng sáng/tối rõ rệt (da mí, lông mi, sclera,
                                  # tròng đen...); hầu hết vật cản (tay, giấy, điện thoại,
                                  # sách, tóc...) áp sát camera có độ tương phản thấp hơn
                                  # nhiều dù bề mặt có hoa văn nhẹ -> kiểm tra này tổng quát
                                  # cho NHIỀU LOẠI vật cản, không riêng gì bàn tay

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path("models/face_landmarker.task")

# Landmark index (trong 478 điểm của FaceLandmarker) dùng để khoanh vùng mắt.
RIGHT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEFT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ============================== TIỆN ÍCH ==============================


def ensure_face_landmarker_model(path: Path = FACE_LANDMARKER_MODEL_PATH) -> str:
    """Tự động tải model FaceLandmarker (.task) nếu chưa có sẵn."""
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
                winsound.Beep(1000, 300)
            else:
                print("\a", end="", flush=True)
            time.sleep(0.4)


def get_landmark_points(landmarks, idx_list, img_w, img_h):
    return np.array([
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in idx_list
    ])


def crop_eye(frame, landmarks, idx_list, img_w, img_h, margin=6):
    pts = get_landmark_points(landmarks, idx_list, img_w, img_h)
    x1, y1 = pts[:, 0].min() - margin, pts[:, 1].min() - margin
    x2, y2 = pts[:, 0].max() + margin, pts[:, 1].max() + margin
    # Nếu vùng mắt bị khuất ra ngoài khung hình (bị vật cản/quay mặt che mất
    # một phần), toạ độ gốc sẽ âm hoặc vượt quá kích thước frame — coi là
    # không lấy được mắt này, KHÔNG cắt xén cho vừa khung (tránh crop bị méo/
    # thiếu, dẫn tới dự đoán sai mà không biết).
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def is_crop_usable(crop_bgr):
    """Kiểm tra crop mắt có dùng được không: tồn tại, đủ sáng, và không bị
    vật cản che (tay, sách, điện thoại, tóc, hoặc bất kỳ vật gì khác áp sát
    camera).

    - crop None -> mắt bị khuất ra khỏi camera / không lấy được toạ độ.
    - Quá tối/quá sáng -> ánh sáng không đủ để nhận diện đáng tin cậy.
    - Edge density quá thấp -> vùng ảnh quá "phẳng", không có chi tiết dạng
      mắt (viền mí, lông mi...).
    - Độ tương phản (std) quá thấp -> không có sự chênh lệch sáng/tối rõ
      rệt như mắt thật (da mí/lông mi/sclera/tròng đen), đặc trưng chung
      của rất nhiều loại vật cản khác nhau (không riêng gì bàn tay) khi áp
      sát camera.
    Hai điều kiện edge density + contrast std bổ trợ cho nhau, giúp tổng
    quát hoá việc phát hiện vật cản thay vì chỉ nhận diện đúng 1 loại.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if not (MIN_EYE_BRIGHTNESS <= mean_brightness <= MAX_EYE_BRIGHTNESS):
        return False

    contrast_std = float(np.std(gray))
    if contrast_std < MIN_CONTRAST_STD:
        return False

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    if edge_density < MIN_EDGE_DENSITY:
        return False

    return True


def predict_eye_state(model, eye_crop_bgr):
    """Trả về (is_open, prob_open, debug_img) hoặc (None, None, None) nếu không crop được.

    Quan trọng: model được train trên ảnh GRAYSCALE (dataset MRL gốc là ảnh
    đen trắng, được nhân 3 kênh giống nhau lúc train). Ảnh màu thật từ webcam
    phải được chuyển về grayscale rồi nhân 3 kênh tương tự thì mới đúng phân
    phối dữ liệu model đã học, tránh dự đoán sai lệch do màu da/ánh sáng.
    """
    if eye_crop_bgr is None or eye_crop_bgr.size == 0:
        return None, None, None
    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(gray_3ch, (EYE_MODEL_IMG_SIZE, EYE_MODEL_IMG_SIZE))
    debug_img = img.copy()  # ảnh thực tế đưa vào model, dùng để hiển thị debug
    img = np.expand_dims(img.astype(np.float32), axis=0)
    prob_open = float(model.predict(img, verbose=0)[0][0])
    return prob_open >= EYE_OPEN_PROB_THRESHOLD, prob_open, debug_img


def overlay_thumbnail(frame, thumb_rgb, x, y, size=100, label=""):
    """Dán ảnh thumbnail (RGB) lên góc frame (BGR) để debug trực quan."""
    if thumb_rgb is None:
        return
    thumb_bgr = cv2.cvtColor(cv2.resize(thumb_rgb, (size, size)), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    if x + size > w or y + size > h:
        return
    frame[y:y + size, x:x + size] = thumb_bgr
    cv2.rectangle(frame, (x, y), (x + size, y + size), (255, 255, 255), 1)
    if label:
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ============================== MAIN ==============================


def main():
    parser = argparse.ArgumentParser(description="Demo realtime eye open/closed")
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

    alert = AlertPlayer()
    closed_since = None
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

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                right_crop = crop_eye(frame, landmarks, RIGHT_EYE_IDX, img_w, img_h)
                left_crop = crop_eye(frame, landmarks, LEFT_EYE_IDX, img_w, img_h)

                # Yêu cầu CẢ 2 mắt đều hợp lệ (crop được + đủ sáng) mới dự
                # đoán. Nếu 1 bên bị vật cản che, bị khuất ra khỏi camera,
                # hoặc ánh sáng không đủ để nhận diện rõ -> không đủ tin cậy
                # để đưa ra kết luận, báo "không nhận diện được khuôn mặt"
                # thay vì đoán liều dựa trên 1 mắt.
                if is_crop_usable(right_crop) and is_crop_usable(left_crop):
                    right_open, right_prob, right_debug = predict_eye_state(model, right_crop)
                    left_open, left_prob, left_debug = predict_eye_state(model, left_crop)

                    is_open = right_open and left_open  # cả 2 mắt cùng mở mới tính là "mở"
                    eye_label = "MO" if is_open else "NHAM"
                else:
                    eye_label = "Khong nhan dien duoc khuon mat"

                overlay_thumbnail(frame, right_debug, img_w - 220, 10, label="Phai (input model)")
                overlay_thumbnail(frame, left_debug, img_w - 110, 10, label="Trai")

            # cập nhật thời gian nhắm mắt liên tục
            if is_open is False:
                if closed_since is None:
                    closed_since = now
            else:
                closed_since = None

            closed_duration = (now - closed_since) if closed_since else 0.0
            is_alert = closed_duration >= EYE_CLOSED_ALERT_SEC

            if is_alert:
                alert.start()
            else:
                alert.stop()

            # ---- Vẽ overlay ----
            if is_alert:
                color = (0, 0, 255)      # đỏ: đang cảnh báo
            elif is_open is None:
                color = (128, 128, 128)  # xám: không nhận diện được
            elif is_open:
                color = (0, 200, 0)      # xanh lá: mắt mở
            else:
                color = (0, 165, 255)    # cam: mắt nhắm (chưa đủ lâu để cảnh báo)
            cv2.putText(frame, f"Mat: {eye_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if right_prob is not None:
                cv2.putText(frame, f"Xac suat mo - Phai: {right_prob:.2f}  Trai: {left_prob:.2f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Nham lien tuc: {closed_duration:.1f}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if is_alert:
                cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1), color, 8)
                cv2.putText(frame, "CANH BAO: DANG NHAM MAT QUA LAU!", (20, img_h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Demo - Eye Open/Closed Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        alert.stop()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
