"""
Xem lại lịch sử các sự kiện đã ghi nhận (đổi trạng thái, ngáp, nhắm mắt kéo dài).

Cách dùng:
    python view_history.py --history_path data/history_log.csv
    python view_history.py --history_path data/history_log.csv --date 2026-09-09
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Xem lịch sử buồn ngủ đã ghi nhận")
    parser.add_argument("--history_path", type=str, default="data/history_log.csv")
    parser.add_argument("--date", type=str, default=None,
                         help="Lọc theo ngày, định dạng YYYY-MM-DD (mặc định: xem tất cả)")
    args = parser.parse_args()

    path = Path(args.history_path)
    if not path.exists():
        print(f"Chưa có file lịch sử tại {path}. Hãy chạy realtime_app.py trước.")
        return

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.date and not row["timestamp"].startswith(args.date):
                continue
            rows.append(row)

    if not rows:
        print("Không có sự kiện nào phù hợp.")
        return

    print(f"{'Thời gian':<20} {'Loại sự kiện':<20} {'Chi tiết'}")
    print("-" * 65)
    for row in rows:
        print(f"{row['timestamp']:<20} {row['event_type']:<20} {row['detail']}")

    counts = Counter(row["event_type"] for row in rows)
    drowsy_transitions = sum(
        1 for row in rows if row["event_type"] == "state_change" and row["detail"].endswith("DROWSY")
    )

    print("\n=== THỐNG KÊ ===")
    print(f"Tổng số sự kiện: {len(rows)}")
    for event_type, n in counts.items():
        print(f"  - {event_type}: {n}")
    print(f"Số lần chuyển sang trạng thái DROWSY: {drowsy_transitions}")


if __name__ == "__main__":
    main()
