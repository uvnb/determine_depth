"""Threaded Video Stream cho Camera.

Tách riêng luồng đọc frame từ phần cứng (I/O) ra khỏi luồng xử lý ảnh.
Giúp đạt tối đa FPS và giảm thiểu trễ khung hình (latency), đặc biệt hiệu quả trên Raspberry Pi 4B.
"""

import threading
import time
import cv2

class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        # Thử mở camera với V4L2 backend trước (chuẩn trên Linux/Raspberry Pi)
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        # Thiết lập codec MJPG nếu có thể để tối ưu tốc độ truyền khung hình
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Khởi động luồng chạy nền đọc camera."""
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera thiết bị {self.src}")

        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.2)  # Đợi camera ổn định
        return self

    def _update(self):
        """Vòng lặp ngầm liên tục cập nhật frame mới nhất."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        """Lấy khung hình mới nhất một cách an toàn đa luồng."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        """Dừng luồng và giải phóng tài nguyên."""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
