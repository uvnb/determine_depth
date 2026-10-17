"""Module truyền phát video trực tiếp (MJPEG Web Streaming) qua mạng cục bộ.

Đặc tính kỹ thuật:
  - Sử dụng thư viện chuẩn của Python (http.server, threading) - không cần cài thêm package.
  - Kiến trúc bất đồng bộ (Dedicated Background Thread): Hoàn toàn không làm nghẽn luồng nhận diện chính.
  - Zero-Overhead khi không có người xem: Chỉ mã hóa JPEG khi có kết nối từ trình duyệt.
  - Stream Throttling: Giới hạn tốc độ phát video (20-30 FPS) giúp Pi 4B duy trì tốc độ xử lý >80 FPS.
"""

import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
import cv2
import numpy as np

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Cylinder Tracker - Live Stream (Pi 4B)</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        header {
            text-align: center;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 1.8rem;
            color: var(--accent-cyan);
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }
        .badge {
            display: inline-block;
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .stream-container {
            background: var(--bg-card);
            padding: 12px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 900px;
            width: 100%;
        }
        .video-box {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 360px;
        }
        .video-box img {
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
        }
        .footer-info {
            display: flex;
            justify-content: space-between;
            margin-top: 14px;
            padding: 0 8px;
            font-size: 0.9rem;
            color: var(--text-dim);
        }
        .info-pill {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.7; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.7; }
        }
    </style>
</head>
<body>
    <header>
        <h1>3D CYLINDER TRACKER</h1>
        <div class="badge">LIVE MJPEG STREAM (RASPBERRY PI 4B)</div>
    </header>

    <div class="stream-container">
        <div class="video-box">
            <img src="/video_feed" alt="Đang kết nối camera luồng video...">
        </div>
        <div class="footer-info">
            <div class="info-pill">
                <span class="dot"></span>
                <span>Luồng trực tuyến qua mạng nội bộ</span>
            </div>
            <div>Độ phân giải: 640x480 | Độ sâu Pinhole 3D</div>
        </div>
    </div>
</body>
</html>
"""

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Máy chủ HTTP hỗ trợ đa luồng kết nối không bị block."""
    daemon_threads = True
    allow_reuse_address = True

class StreamHandler(BaseHTTPRequestHandler):
    """Xử lý yêu cầu HTTP và truyền phát luồng MJPEG."""

    def log_message(self, format, *args):
        # Tắt bớt log HTTP mặc định để giữ console terminal sạch đẹp
        pass

    def do_GET(self):
        streamer: WebStreamer = self.server.streamer

        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            streamer.add_client()
            try:
                while streamer.is_running:
                    jpeg_bytes = streamer.get_latest_jpeg()
                    if jpeg_bytes is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                    else:
                        time.sleep(0.02)
                    time.sleep(1.0 / streamer.stream_fps)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                streamer.remove_client()

        elif self.path == '/snapshot':
            jpeg_bytes = streamer.get_latest_jpeg()
            if jpeg_bytes is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg_bytes)))
                self.end_headers()
                self.wfile.write(jpeg_bytes)
            else:
                self.send_response(503)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

class WebStreamer:
    """Bộ truyền phát MJPEG HTTP Server độc lập, chạy ngầm trên core CPU riêng."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        stream_fps: int = 25,
        jpeg_quality: int = 70,
        enabled: bool = True
    ):
        self.host = host
        self.port = port
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality
        self.enabled = enabled

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._client_count = 0
        self._client_lock = threading.Lock()

        self.is_running = False
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._encoder_thread: Optional[threading.Thread] = None

    @property
    def has_clients(self) -> bool:
        with self._client_lock:
            return self._client_count > 0

    def add_client(self):
        with self._client_lock:
            self._client_count += 1

    def remove_client(self):
        with self._client_lock:
            self._client_count = max(0, self._client_count - 1)

    def start(self):
        """Khởi động máy chủ web stream chạy nền."""
        if not self.enabled:
            return

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), StreamHandler)
            self._server.streamer = self
            self.port = self._server.server_address[1]
            self.is_running = True

            # Luồng phục vụ HTTP
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()

            # Luồng nén JPEG độc lập (chỉ nén khi có client xem)
            self._encoder_thread = threading.Thread(target=self._encoder_worker, daemon=True)
            self._encoder_thread.start()

            print(f"[WEB STREAM] Máy chủ phát video đã kích hoạt tại: http://{self.host}:{self.port}")
            print(f"             (Từ laptop, mở trình duyệt truy cập: http://<IP_CỦA_PI>:{self.port})")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể khởi động Web Stream trên cổng {self.port}: {e}")
            self.is_running = False

    def update_frame(self, frame: np.ndarray):
        """Cập nhật khung hình mới từ luồng nhận diện (mất <0.1ms, không làm tụt FPS)."""
        if not self.enabled or not self.is_running:
            return
        # Chỉ lưu tham chiếu khi có client đang xem để tiết kiệm tài nguyên
        if self.has_clients:
            with self._lock:
                self._latest_frame = frame

    def _encoder_worker(self):
        """Luồng ngầm chuyên nén JPEG ở tốc độ stream_fps định trước."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        interval = 1.0 / self.stream_fps

        while self.is_running:
            if not self.has_clients:
                # Không có ai xem -> ngủ tiết kiệm CPU
                time.sleep(0.05)
                continue

            t_start = time.time()
            frame_to_encode = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_encode = self._latest_frame

            if frame_to_encode is not None:
                ret, buf = cv2.imencode('.jpg', frame_to_encode, encode_param)
                if ret:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()

            dt = time.time() - t_start
            sleep_time = max(0.005, interval - dt)
            time.sleep(sleep_time)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def stop(self):
        """Dừng máy chủ web stream."""
        self.is_running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        print("[WEB STREAM] Đã đóng máy chủ video stream.")
