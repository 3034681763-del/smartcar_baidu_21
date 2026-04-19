import copy
import os
import struct
import time
from multiprocessing import shared_memory

import cv2
import numpy as np
import pyudev
from paddlelite.lite import MobileConfig, create_paddle_predictor


class CameraCapture:
    def __init__(self, lane_camera_path, task_camera_path):
        self.lane_camera_path = lane_camera_path
        self.task_camera_path = task_camera_path
        self.shm_lane_image = None
        self.shm_task_image = None
        self.shm_lane_ret_bytes = None
        self.lane_cap = None
        self.task_cap = None
        self.paddle_predictor = None
        self.model_path = None

    def find_camera_by_path(self, physical_path):
        if isinstance(physical_path, str) and physical_path.startswith("/dev/video"):
            return physical_path

        context = pyudev.Context()
        for device in context.list_devices(subsystem="video4linux"):
            end_index = device.device_path.rfind("/video4linux")
            cleaned_string = device.device_path[:end_index] if end_index != -1 else device.device_path
            if cleaned_string.endswith(physical_path):
                return "/dev/" + device.device_path.rsplit("/", 1)[1]
        return None

    def setup_shared_memory(self, lane_shm_name, task_shm_name, shm_lane_name):
        self.shm_lane_image = shared_memory.SharedMemory(name=lane_shm_name)
        self.shm_task_image = shared_memory.SharedMemory(name=task_shm_name)
        self.shm_lane_ret_bytes = shared_memory.SharedMemory(name=shm_lane_name)

    def setup_paddle_predictor(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CamCapture] Lane model file not found: {model_path}")

        try:
            config = MobileConfig()
            config.set_model_from_file(model_path)
            self.paddle_predictor = create_paddle_predictor(config)
            print(f"[CamCapture] Lane model loaded: {model_path}")
        except Exception as exc:
            raise RuntimeError(
                f"[CamCapture] Failed to load lane model: {model_path}. Reason: {exc}"
            ) from exc

    def open_single_camera(self, physical_path, role_name):
        device = self.find_camera_by_path(physical_path)
        if device is None:
            print(f"[CamCapture] {role_name} camera not found: {physical_path}")
            return None

        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            print(f"[CamCapture] Failed to open {role_name} camera with V4L2 backend: {device}")
            capture = cv2.VideoCapture(device)
        if not capture.isOpened():
            print(f"[CamCapture] Failed to open {role_name} camera: {device}")
            return None

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 60)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        actual_codec = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))

        print(f"[CamCapture] {role_name} camera ready: {device}")
        print(
            f"[CamCapture] {role_name} camera config: {actual_width}x{actual_height} "
            f"@ {actual_fps:.2f} FPS, codec={actual_codec}"
        )
        return capture

    def open_camera(self):
        self.lane_cap = self.open_single_camera(self.lane_camera_path, "Lane")
        if self.lane_cap is None:
            return False

        self.task_cap = self.open_single_camera(self.task_camera_path, "Task")
        if self.task_cap is None:
            self.lane_cap.release()
            self.lane_cap = None
            return False

        return True

    def process_frame(self, frame, shm_image_array):
        frame_copy = copy.deepcopy(frame)
        shm_image_array[:] = np.array(frame_copy)[:]

    def run_inference(self, frame):
        if self.paddle_predictor is None:
            raise RuntimeError(
                f"[CamCapture] Lane model is not initialized. Current path: {self.model_path}"
            )

        img = cv2.resize(frame, (128, 128))
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = img[:, :, ::-1].astype("float32")
        img = img.transpose((2, 0, 1))
        image_data = img[np.newaxis, :]

        input_tensor = self.paddle_predictor.get_input(0)
        input_tensor.from_numpy(image_data)
        self.paddle_predictor.run()
        return self.paddle_predictor.get_output(0).numpy()

    def draw_overlay(self, frame, deviation, infer_speed, fps):
        display = frame.copy()
        cv2.putText(
            display,
            f"Deviation: {deviation:.2f}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"InferSpeed: {infer_speed:.2f}",
            (16, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"FPS: {fps:.2f}",
            (16, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Press q to quit",
            (16, 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def run(self, stop_event):
        lane_window_name = "Lane Preview"
        task_window_name = "Task Preview"
        cv2.namedWindow(lane_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(task_window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        fps_timer = time.time()
        current_fps = 0.0

        print("[CamCapture] Lane/task camera loop started")
        try:
            while not stop_event.is_set():
                lane_ok, lane_frame = self.lane_cap.read()
                task_ok, task_frame = self.task_cap.read()
                if not lane_ok:
                    continue

                frame_count += 1
                elapsed_time = time.time() - fps_timer
                if elapsed_time >= 1.0:
                    current_fps = frame_count / elapsed_time
                    print(f"[CamCapture] Inference FPS: {current_fps:.2f}")
                    frame_count = 0
                    fps_timer = time.time()

                lane_frame = cv2.resize(lane_frame, (640, 640), interpolation=cv2.INTER_AREA)
                lane_infer_ret = self.run_inference(lane_frame)
                deviation = float(lane_infer_ret[0][0]) * 1.1
                infer_speed = float(lane_infer_ret[0][1])

                ret_bytes = struct.pack("=ff", deviation, infer_speed)
                self.shm_lane_ret_bytes.buf[0:len(ret_bytes)] = ret_bytes
                self.process_frame(
                    lane_frame,
                    np.ndarray(lane_frame.shape, dtype=lane_frame.dtype, buffer=self.shm_lane_image.buf),
                )

                if task_ok:
                    task_frame = cv2.resize(task_frame, (640, 640), interpolation=cv2.INTER_AREA)
                    self.process_frame(
                        task_frame,
                        np.ndarray(task_frame.shape, dtype=task_frame.dtype, buffer=self.shm_task_image.buf),
                    )
                    cv2.imshow(task_window_name, task_frame)

                lane_preview = self.draw_overlay(lane_frame, deviation, infer_speed, current_fps)
                cv2.imshow(lane_window_name, lane_preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[CamCapture] q pressed, stopping framework")
                    stop_event.set()
                    break
        finally:
            print("[CamCapture] Stopping camera process")
            if self.lane_cap is not None:
                self.lane_cap.release()
            if self.task_cap is not None:
                self.task_cap.release()
            cv2.destroyAllWindows()


def vidpub_course(
    stop_event,
    lane_shm_key,
    task_shm_key,
    shm_lane_key,
    model_path="src/cnn_auto.nb",
    lane_camera_path="1-2.2:1.0",
    task_camera_path="1-2.4:1.0",
):
    camera_capture = CameraCapture(
        lane_camera_path=lane_camera_path,
        task_camera_path=task_camera_path,
    )
    camera_capture.setup_shared_memory(
        lane_shm_name=lane_shm_key,
        task_shm_name=task_shm_key,
        shm_lane_name=shm_lane_key,
    )

    try:
        camera_capture.setup_paddle_predictor(model_path=model_path)
    except Exception as exc:
        print(exc)
        return

    if camera_capture.open_camera():
        camera_capture.run(stop_event)
