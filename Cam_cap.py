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
    def __init__(self, camera_path_0):
        self.camera_path_0 = camera_path_0
        self.shm_image_0 = None
        self.shm_lane_ret_bytes = None
        self.cap_0 = None
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

    def setup_shared_memory(self, shm_img_name, shm_lane_name):
        self.shm_image_0 = shared_memory.SharedMemory(name=shm_img_name)
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

    def open_camera(self):
        dev_0 = self.find_camera_by_path(self.camera_path_0)
        if dev_0 is None:
            print(f"[CamCapture] Camera not found: {self.camera_path_0}")
            return False

        self.cap_0 = cv2.VideoCapture(dev_0)
        if not self.cap_0.isOpened():
            print(f"[CamCapture] Failed to open camera: {dev_0}")
            return False

        self.cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[CamCapture] Camera ready: {dev_0}")
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
        window_name = "Lane Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        fps_timer = time.time()
        current_fps = 0.0

        print("[CamCapture] Lane camera and inference loop started")
        try:
            while not stop_event.is_set():
                ret_0, frame_0 = self.cap_0.read()
                if not ret_0:
                    continue

                frame_count += 1
                elapsed_time = time.time() - fps_timer
                if elapsed_time >= 1.0:
                    current_fps = frame_count / elapsed_time
                    print(f"[CamCapture] Inference FPS: {current_fps:.2f}")
                    frame_count = 0
                    fps_timer = time.time()

                frame_0 = cv2.resize(frame_0, (640, 640), interpolation=cv2.INTER_AREA)
                lane_infer_ret = self.run_inference(frame_0)
                deviation = float(lane_infer_ret[0][0]) * 1.1
                infer_speed = float(lane_infer_ret[0][1])

                ret_bytes = struct.pack("=ff", deviation, infer_speed)
                self.shm_lane_ret_bytes.buf[0:len(ret_bytes)] = ret_bytes
                self.process_frame(
                    frame_0,
                    np.ndarray(frame_0.shape, dtype=frame_0.dtype, buffer=self.shm_image_0.buf),
                )

                preview = self.draw_overlay(frame_0, deviation, infer_speed, current_fps)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[CamCapture] q pressed, stopping framework")
                    stop_event.set()
                    break
        finally:
            print("[CamCapture] Stopping camera process")
            if self.cap_0 is not None:
                self.cap_0.release()
            cv2.destroyAllWindows()


def vidpub_course(stop_event, shm0, shm_lane, model_path="src/cnn_auto.nb", camera_path="1-2.2:1.0"):
    camera_capture = CameraCapture(camera_path_0=camera_path)
    camera_capture.setup_shared_memory(shm_img_name=shm0, shm_lane_name=shm_lane)

    try:
        camera_capture.setup_paddle_predictor(model_path=model_path)
    except Exception as exc:
        print(exc)
        return

    if camera_capture.open_camera():
        camera_capture.run(stop_event)
