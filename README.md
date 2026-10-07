# DATN — Hệ thống phát hiện & cảnh báo buồn ngủ tài xế

## Bước hiện tại: Chuẩn bị dataset

### 1. Dataset đã chọn: MRL Eye Dataset

Lý do chọn:
- Dataset chuẩn, được nhiều đề tài/paper cùng chủ đề sử dụng (kể cả model
  `open-closed-eye-0001` của Intel OpenVINO cũng train trên dataset này,
  đạt ~95.8% accuracy).
- Đã là ảnh mắt được crop sẵn (không cần tự làm face/eye detection cho bước
  train model) → tập trung vào phần ML.
- Có gán nhãn sẵn: trạng thái mắt (open/closed), có/không đeo kính, điều
  kiện ánh sáng, phản chiếu — rất hữu ích cho phần đánh giá hạn chế
  (ánh sáng yếu, đeo kính) mà đề tài yêu cầu.
- ~84,898 ảnh từ 37 người → đủ lớn để train, đồng thời có subject ID rõ
  ràng để tách train/test theo người (đúng yêu cầu trong đề tài: "tránh
  ảnh của cùng một người vừa nằm trong train vừa nằm trong test").

### 2. Cách tải dataset

Trang chủ dataset (cần tải thủ công vì máy chủ tải file không nằm trong
danh sách domain tôi được phép truy cập):

```
http://mrl.cs.vsb.cz/eyedataset.html
```

Link tải trực tiếp (chạy trên máy của bạn):

```bash
wget http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip
unzip mrlEyes_2018_01.zip -d data/raw/
```

Dataset sau khi giải nén có cấu trúc: `data/raw/mrlEyes_2018_01/s0001/`,
`s0002/`, ... (mỗi thư mục là 1 người), bên trong là các ảnh `.png` được
đặt tên theo quy tắc:

```
s<subjectID>_<imageID>_<gender>_<glasses>_<eyestate>_<reflections>_<lighting>_<sensorID>.png
```

Ví dụ: `s0001_00001_0_0_0_0_0_01.png`

Trường quan trọng nhất là `<eyestate>`: `0 = closed`, `1 = open`.

Nếu không tải được từ trang gốc, có thể tìm mirror trên Kaggle (tìm
"MRL Eye Dataset" trên kaggle.com) — cấu trúc file có thể hơi khác, cần
kiểm tra lại trước khi chạy script.

### 3. Chạy tiền xử lý

```bash
pip install -r requirements.txt
python src/prepare_dataset.py --raw_dir data/raw/mrlEyes_2018_01 --out_dir data/processed --img_size 64
```

Script sẽ:
- Parse nhãn từ tên file (không cần file annotation riêng)
- Chia train/val/test theo **subject ID** (mặc định 70/15/15 số người,
  không phải số ảnh) để tránh rò rỉ dữ liệu giữa các tập
- Resize ảnh về kích thước vuông (mặc định 64x64), convert grayscale
- Lưu vào `data/processed/{train,val,test}/{open,closed}/`
- In thống kê số lượng ảnh mỗi lớp, mỗi tập

### 4. Sau bước này

Khi dataset đã sẵn sàng trong `data/processed/`, bước tiếp theo là xây
CNN đơn giản (hoặc transfer learning) để phân loại open/closed — nói với
tôi khi bạn đã có dữ liệu để tiếp tục.

## Bước: Train model (đã hoàn thành)

```bash
python src/train_model.py --data_dir data/processed --out_dir models --epochs 15
python src/evaluate_model.py --model_path models/eye_state_model.keras --data_dir data/processed
```

Kết quả đã đạt: Accuracy ~93.7%, F1 lớp closed ~0.96, F1 lớp open ~0.82
trên tập test.

## Bước: Ứng dụng realtime (webcam + cảnh báo + lịch sử)

### Chạy ứng dụng

```bash
python src/realtime_app.py --model_path models/eye_state_model.keras
```

Nhấn `q` trong cửa sổ video để thoát.

### Cách hoạt động

1. **MediaPipe FaceMesh** phát hiện 468 landmark khuôn mặt mỗi frame.
2. Vùng mắt trái/phải được cắt ra từ landmark, đưa vào **model CNN**
   (đã train ở bước trước) để phân loại open/closed. Nếu ít nhất 1 trong
   2 mắt được nhận là "open" thì coi là mắt mở (giảm nhiễu khi 1 mắt bị
   che góc quay).
3. Vùng miệng dùng để tính **MAR (Mouth Aspect Ratio)** = khoảng cách dọc
   / khoảng cách ngang của miệng — MAR cao kéo dài liên tục vài frame
   được tính là 1 lần ngáp (không dùng model riêng, đúng yêu cầu đề tài).
4. **Thuật toán buồn ngủ** (`DrowsinessMonitor` trong `realtime_app.py`)
   kết hợp:
   - Thời gian mắt nhắm liên tục (không kết luận từ 1 frame đơn lẻ)
   - Số lần ngáp trong 60 giây gần nhất
   - Phân 3 mức: `NORMAL` → `WARNING` → `DROWSY`
5. Khi vào trạng thái `DROWSY`: phát âm thanh cảnh báo liên tục (dùng
   `winsound`, không cần cài thêm) + viền đỏ quanh khung hình. Cảnh báo
   tự dừng ngay khi trạng thái quay lại `NORMAL`/`WARNING`.
6. Mọi sự kiện (đổi trạng thái, phát hiện ngáp, đợt nhắm mắt kéo dài)
   được ghi vào `data/history_log.csv` kèm timestamp.

### Các ngưỡng có thể tinh chỉnh

Ở đầu file `src/realtime_app.py`, phần `CẤU HÌNH`:
- `EYE_WARNING_SEC` / `EYE_DROWSY_SEC`: thời gian nhắm mắt liên tục để
  chuyển mức (mặc định 1.0s / 2.0s)
- `MAR_THRESHOLD`: ngưỡng tỉ lệ mở miệng để tính là đang ngáp (mặc định
  0.55 — nên tự căn chỉnh theo camera/khuôn mặt thực tế của bạn, quan sát
  giá trị MAR hiển thị trên màn hình khi test)
- `YAWN_WARNING_COUNT` / `YAWN_DROWSY_COUNT`: số lần ngáp trong 60s để
  chuyển mức

Nên thử nghiệm, ghi lại các giá trị đã chọn và lý do vào báo cáo — đây
cũng là một phần đánh giá của đề tài.

### Xem lại lịch sử

```bash
python src/view_history.py --history_path data/history_log.csv
```

In toàn bộ danh sách sự kiện kèm thống kê tổng số lần buồn ngủ, ngáp,
đợt nhắm mắt kéo dài. Có thể lọc theo ngày với `--date 2026-09-09`.

## Lưu ý quan trọng: đừng xóa thư mục cũ khi cập nhật code

Từ giờ, **KHÔNG xóa thư mục `datn-drowsy-detection` cũ rồi giải nén lại** —
làm vậy sẽ xóa mất `data/processed/` (dataset đã xử lý) và `models/`
(model đã train, tốn ~30 phút để train lại). Khi nhận file mới, giải nén
vào một thư mục tạm riêng, rồi chỉ **copy đè các file `.py` đã thay đổi**
trong `src/` vào đúng thư mục project đang dùng — giữ nguyên `data/` và
`models/` hiện có.

Nếu lỡ mất `data/processed/` hoặc `models/`:
- `data/processed/` mất nhưng `mrlEyes_2018_01` (dataset gốc) còn nguyên
  → chạy lại `prepare_dataset.py` là có lại ngay (vài phút)
- `models/*.keras` mất → phải train lại từ đầu bằng `train_model.py`
  (~30 phút)

## Bước: Thu thập dữ liệu mắt của chính bạn để cải thiện model

Vấn đề gặp phải: model train trên dataset MRL (ảnh chụp camera cận hồng
ngoại, điều kiện chuẩn hoá) dự đoán sai khi test trên webcam thực tế do
**domain gap** — ảnh thực tế khác nhiều so với ảnh lúc train. Cách khắc
phục hiệu quả nhất: bổ sung chính ảnh mắt của bạn (chụp từ webcam bạn sẽ
dùng để demo) vào tập train, rồi fine-tune tiếp model đã có.

### 1. Thu thập dữ liệu

```bash
python src/collect_custom_eye_data.py --out_dir data/custom
```

Phím điều khiển: `o` = lưu frame hiện tại là **open**, `c` = lưu là
**closed**, `q` = thoát. Mỗi lần bấm sẽ lưu cả 2 mắt (nếu crop được).

Cách thu hiệu quả:
- Thu cả mở bình thường/mở to/nheo mắt, nhắm hờ/nhắm chặt
- Đổi góc mặt, khoảng cách tới camera, ánh sáng nếu có thể (phòng
  sáng/tối) — vừa tăng đa dạng, vừa phục vụ luôn phần đánh giá hạn chế
  của đề tài
- Nên thu tối thiểu ~150-200 ảnh mỗi lớp

Ảnh được lưu vào `data/custom/open/` và `data/custom/closed/` (grayscale,
tự động crop bằng MediaPipe FaceLandmarker) — **chưa** nằm trong
`data/processed/`, cần bước gộp bên dưới.

### 2. Gộp vào tập train

```bash
python src/merge_custom_data.py --custom_dir data/custom --processed_dir data/processed
```

Chỉ copy vào `data/processed/train/`, **giữ nguyên** `val/` và `test/` —
để việc đánh giá sau này vẫn khách quan, độc lập với dữ liệu mới thêm.

### 3. Fine-tune tiếp model đã có

```bash
python src/finetune_custom.py --base_model_path models/eye_state_model.keras --data_dir data/processed --out_dir models --epochs 5
```

Nạp lại model đã train (`models/eye_state_model.keras`), tiếp tục train
thêm vài epoch với learning rate rất nhỏ (`1e-5` mặc định) — đủ để model
thích nghi với đặc điểm webcam/khuôn mặt của bạn mà không phá vỡ những gì
đã học từ tập MRL gốc. Kết quả lưu tại
`models/eye_state_model_finetuned.keras`.

### 4. Đánh giá lại

```bash
python src/evaluate_model.py --model_path models/eye_state_model_finetuned.keras --data_dir data/processed
```

So sánh với kết quả trước khi fine-tune (Accuracy 93.68%, F1 open 0.82)
để xem có cải thiện không — nên đưa cả 2 kết quả (trước/sau khi bổ sung
dữ liệu thực tế) vào báo cáo, đây là một điểm thảo luận hay cho đồ án.

### Sau đó

Test lại bằng `test_model_grayscale.py` hoặc `demo_eye_realtime.py`
(nhớ đổi `--model_path` sang bản `_finetuned.keras` mới) để xem độ chính
xác có cải thiện không trên chính webcam của bạn.

Đề tài yêu cầu test thêm các trường hợp: ánh sáng yếu, đeo kính, quay
mặt. Hãy chạy `realtime_app.py` thử từng điều kiện này, quan sát và ghi
lại nhận xét (model có còn nhận diện đúng không, MAR có ổn định không)
để đưa vào phần đánh giá hạn chế của báo cáo.
