"""Chương trình chính nhận diện khối trụ và hiển thị tọa độ 3D (X, Y, Z) thời gian thực.

Tích hợp:
  - Tối ưu đa luồng camera (Threaded Video Stream).
  - Khử méo thấu kính (Undistortion) từ file hiệu chuẩn.
  - Phân đoạn ảnh và trích xuất đặc trưng 2D.
  - Ước lượng tọa độ không gian 3D Pinhole.
  - Bộ lọc làm mịn tọa độ (Temporal Smoothing: Moving Average / Kalman).
  - Giao diện HUD chuyên nghiệp, hiển thị FPS và trạng thái hệ thống.
"""

import os
import sys
import time
import cv2
import numpy as np
import yaml

from core.camera_stream import CameraStream
from core.cylinder_detector import CylinderDetector
from core.pinhole_3d import Pinhole3DEstimator
from core.tracker_filter import MovingAverageFilter3D, KalmanFilter3D
from core.serial_comm import SerialSender

def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "configs", "config.yaml")

    if not os.path.exists(cfg_path):
        print(f"[CẢNH BÁO] Không tìm thấy file {cfg_path}, sử dụng cấu hình mặc định.")
        return {}

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_calibration(calib_file_rel, base_dir):
    calib_path = os.path.join(base_dir, calib_file_rel)
    if not os.path.exists(calib_path):
        print(f"[THÔNG BÁO] Chưa có file hiệu chuẩn {calib_path}. Sử dụng thông số Pinhole xấp xỉ.")
        return None, None, False

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam_mat = np.array(data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float32)
        is_calibrated = data.get("is_calibrated", False)
        return cam_mat, dist_coeffs, is_calibrated
    except Exception as e:
        print(f"[LỖI] Đọc file calibration thất bại: {e}")
        return None, None, False

def draw_hud(frame, detection, coords_3d, distance, fps, is_calibrated, filter_enabled, show_pip_mask=True, serial_sender=None):
    """Vẽ giao diện hiển thị thông tin thực tế (HUD) lên khung hình."""
    h, w = frame.shape[:2]

    # Thanh header trên cùng
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    calib_str = "CALIBRATED" if is_calibrated else "DEFAULT INTRINSICS"
    calib_color = (0, 255, 0) if is_calibrated else (0, 165, 255)
    cv2.putText(frame, f"3D CYLINDER TRACKER | {calib_str}", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)

    # Hiển thị trạng thái UART Serial nếu có
    if serial_sender and serial_sender.enabled:
        u_text = f"UART: {'CONN' if serial_sender.is_connected else 'NO USB'}"
        u_color = (0, 255, 0) if serial_sender.is_connected else (0, 0, 255)
        cv2.putText(frame, u_text, (w - 240, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, u_color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 95, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if detection["found"]:
        u, v = int(detection["u"]), int(detection["v"])
        bx, by, bw, bh = detection["bbox"]
        X, Y, Z = coords_3d

        # 1. Vẽ khung bao và tâm đối tượng
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)
        cv2.drawMarker(frame, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

        # 2. Bảng thông số nổi bên cạnh vật thể
        panel_x = min(bx + bw + 15, w - 240)
        panel_y = max(by, 50)
        p_w, p_h = 230, 125

        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + p_w, panel_y + p_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + p_w, panel_y + p_h), (0, 255, 0), 1)

        # Nội dung tọa độ 3D
        cv2.putText(frame, f"X : {X:+.3f} m ({X*100:+.1f} cm)", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"Y : {Y:+.3f} m ({Y*100:+.1f} cm)", (panel_x + 10, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"Z : {Z:.3f} m ({Z*100:.1f} cm)", (panel_x + 10, panel_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Dist: {distance:.3f} m | w_px: {detection['w_px']:.0f}", (panel_x + 10, panel_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    else:
        # Thông báo khi chưa tìm thấy vật thể
        cv2.putText(frame, "STATUS: DANG TIM KHOI TRU TREN BANG TRANG...", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3. Góc dưới: Hướng dẫn phím tắt
    filter_status = "ON" if filter_enabled else "OFF"
    uart_status = "ON" if (serial_sender and serial_sender.enabled) else "OFF"
    help_text = f"[q]: Thoat | [u]: Khử méo | [f]: Lọc ({filter_status}) | [t]: UART ({uart_status}) | [s]: Chụp"
    cv2.putText(frame, help_text, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 4. Ảnh nhỏ Picture-in-Picture góc phải dưới (Mask nhị phân)
    if show_pip_mask and detection.get("mask") is not None:
        mask = detection["mask"]
        pip_w, pip_h = 160, 120
        pip_mask = cv2.resize(mask, (pip_w, pip_h))
        pip_bgr = cv2.cvtColor(pip_mask, cv2.COLOR_GRAY2BGR)

        margin = 15
        x1 = w - pip_w - margin
        y1 = h - pip_h - 35
        x2 = x1 + pip_w
        y2 = y1 + pip_h

        frame[y1:y2, x1:x2] = pip_bgr
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, "Mask PIP", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config()

    cam_cfg = cfg.get("camera", {})
    calib_cfg = cfg.get("calibration", {})
    obj_cfg = cfg.get("target_object", {})
    seg_cfg = cfg.get("segmentation", {})
    filter_cfg = cfg.get("filter", {})
    serial_cfg = cfg.get("serial", {})

    dev_idx = cam_cfg.get("device_index", 0)
    width = cam_cfg.get("frame_width", 640)
    height = cam_cfg.get("frame_height", 480)
    fps_target = cam_cfg.get("fps", 30)
    use_threaded = cam_cfg.get("use_threaded_stream", True)

    d_real = float(obj_cfg.get("d_real_m", 0.05))
    min_area = float(obj_cfg.get("min_contour_area", 300))

    # Tải thông số hiệu chuẩn camera
    calib_file = calib_cfg.get("calibration_file", "configs/camera_calibration.yaml")
    cam_matrix, dist_coeffs, is_calibrated = load_calibration(calib_file, base_dir)

    if cam_matrix is not None:
        fx = float(cam_matrix[0, 0])
        fy = float(cam_matrix[1, 1])
        cx = float(cam_matrix[0, 2])
        cy = float(cam_matrix[1, 2])
    else:
        fx, fy = 600.0, 600.0
        cx, cy = width / 2.0, height / 2.0

    print("=" * 60)
    print(" HỆ THỐNG XÁC ĐỊNH TỌA ĐỘ 3D KHỐI TRỤ (OPENCV + PINHOLE)")
    print("=" * 60)
    print(f"- Camera Device: {dev_idx} ({width}x{height})")
    print(f"- Đường kính khối trụ D_real: {d_real} m ({d_real*100:.1f} cm)")
    print(f"- Thông số Pinhole: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"- Trạng thái hiệu chuẩn: {'ĐÃ HIỆU CHUẨN' if is_calibrated else 'MẶC ĐỊNH'}\n")

    # Khởi tạo module Serial UART (USB)
    serial_sender = SerialSender(
        port=serial_cfg.get("port", "/dev/ttyUSB0"),
        baudrate=serial_cfg.get("baudrate", 115200),
        enabled=serial_cfg.get("enabled", False),
        data_format=serial_cfg.get("format", "string"),
        send_rate_hz=serial_cfg.get("send_rate_hz", 30)
    )

    # Khởi tạo các module
    color_mode = seg_cfg.get("color_mode", "blue")
    detector = CylinderDetector(
        color_mode=color_mode,
        blue_lower=tuple(seg_cfg.get("blue_lower", [95, 90, 50])),
        blue_upper=tuple(seg_cfg.get("blue_upper", [130, 255, 255])),
        hsv_lower=tuple(seg_cfg.get("hsv_lower", [95, 90, 50])),
        hsv_upper=tuple(seg_cfg.get("hsv_upper", [130, 255, 255])),
        red_lower1=tuple(seg_cfg.get("red_lower1", [0, 70, 50])),
        red_upper1=tuple(seg_cfg.get("red_upper1", [10, 255, 255])),
        red_lower2=tuple(seg_cfg.get("red_lower2", [165, 70, 50])),
        red_upper2=tuple(seg_cfg.get("red_upper2", [180, 255, 255])),
        min_area=min_area,
        blur_kernel=seg_cfg.get("blur_kernel", 5),
        morph_kernel=seg_cfg.get("morph_kernel", 5),
        camera_matrix=cam_matrix,
        dist_coeffs=dist_coeffs
    )

    estimator_3d = Pinhole3DEstimator(fx=fx, fy=fy, cx=cx, cy=cy, d_real_m=d_real)

    # Bộ lọc làm mịn
    filter_enabled = filter_cfg.get("enabled", True)
    filter_type = filter_cfg.get("filter_type", "moving_average")
    if filter_type == "kalman":
        smooth_filter = KalmanFilter3D(
            process_noise=filter_cfg.get("process_noise", 0.01),
            measurement_noise=filter_cfg.get("measurement_noise", 0.05)
        )
    else:
        smooth_filter = MovingAverageFilter3D(window_size=filter_cfg.get("window_size", 5))

    # Khởi tạo camera
    if use_threaded:
        stream = CameraStream(src=dev_idx, width=width, height=height, fps=fps_target).start()
    else:
        stream = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
        if not stream.isOpened():
            stream = cv2.VideoCapture(dev_idx)
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "3D Cylinder Estimation (OpenCV Pinhole)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps_smooth = 30.0
    show_pip_mask = True
    snapshot_idx = 0

    try:
        while True:
            t_start = time.time()
            if use_threaded:
                ret, frame = stream.read()
            else:
                ret, frame = stream.read()

            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Nhận diện khối trụ
            detection = detector.detect(frame)
            processed_frame = detection["processed_frame"]

            if detection["found"]:
                u = detection["u"]
                v = detection["v"]
                w_px = detection["w_px"]

                raw_X, raw_Y, raw_Z = estimator_3d.estimate(u, v, w_px)

                if filter_enabled:
                    X, Y, Z = smooth_filter.update(raw_X, raw_Y, raw_Z)
                else:
                    X, Y, Z = raw_X, raw_Y, raw_Z

                distance = np.sqrt(X**2 + Y**2 + Z**2)
                coords_3d = (X, Y, Z)
            else:
                smooth_filter.notify_lost()
                coords_3d = (0.0, 0.0, 0.0)
                distance = 0.0

            # Gửi dữ liệu tọa độ qua Serial UART tới vi điều khiển
            serial_sender.send(detection["found"], coords_3d[0], coords_3d[1], coords_3d[2], distance)

            # Tính toán FPS
            dt = time.time() - prev_time
            prev_time = time.time()
            if dt > 0:
                cur_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * cur_fps

            # Vẽ HUD
            draw_hud(
                processed_frame,
                detection,
                coords_3d,
                distance,
                fps_smooth,
                is_calibrated,
                filter_enabled,
                show_pip_mask=show_pip_mask,
                serial_sender=serial_sender
            )

            cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('u'):
                detector.undistort_enabled = not detector.undistort_enabled
                print(f"[STATUS] Khử méo thấu kính: {'BẬT' if detector.undistort_enabled else 'TẮT'}")
            elif key == ord('f'):
                filter_enabled = not filter_enabled
                smooth_filter.reset()
                print(f"[STATUS] Bộ lọc làm mịn: {'BẬT' if filter_enabled else 'TẮT'}")
            elif key == ord('t'):
                serial_sender.enabled = not serial_sender.enabled
                if serial_sender.enabled and not serial_sender.is_connected:
                    serial_sender.connect()
                print(f"[STATUS] Cổng Serial UART: {'BẬT' if serial_sender.enabled else 'TẮT'}")
            elif key == ord('m'):
                show_pip_mask = not show_pip_mask
            elif key == ord('s'):
                snapshot_idx += 1
                snap_path = f"snapshot_{snapshot_idx:03d}.jpg"
                cv2.imwrite(snap_path, processed_frame)
                print(f"[THÀNH CÔNG] Đã lưu snapshot: {snap_path}")

    finally:
        serial_sender.close()
        if use_threaded:
            stream.stop()
        else:
            stream.release()
        cv2.destroyAllWindows()
        print("\nĐã đóng chương trình an toàn.")

if __name__ == "__main__":
    main()
