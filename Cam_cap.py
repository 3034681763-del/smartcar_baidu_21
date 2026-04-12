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
            raise FileNotFoundError(f"[CamCapture] 巡线模型文件不存在: {model_path}")

        try:
            config = MobileConfig()
            config.set_model_from_file(model_path)
            self.paddle_predictor = create_paddle_predictor(config)
            print(f"[CamCapture] 巡线模型加载成功: {model_path}")
        except Exception as exc:
            raise RuntimeError(
                f"[CamCapture] 巡线模型加载失败: {model_path}. 原因: {exc}"
            ) from exc

    def open_camera(self):
        dev_0 = self.find_camera_by_path(self.camera_path_0)
        if dev_0 is None:
            print(f"[CamCapture] 未找到摄像头: {self.camera_path_0}")
            return False

        self.cap_0 = cv2.VideoCapture(dev_0)
        if not self.cap_0.isOpened():
            print(f"[CamCapture] 无法打开摄像头: {dev_0}")
            return False

        self.cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[CamCapture] 摄像头就绪: {dev_0}")
        return True

    def process_frame(self, frame, shm_image_array):
        frame_copy = copy.deepcopy(frame)
        shm_image_array[:] = np.array(frame_copy)[:]

    def run_inference(self, frame):
        if self.paddle_predictor is None:
            raise RuntimeError(
                f"[CamCapture] 巡线模型未初始化，拒绝空载运行。当前模型路径: {self.model_path}"
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

    def run(self, stop_event):
        start_time = time.time()
        frame_count = 0

        print("[CamCapture] 巡线摄像头与推理循环已启动")
        while not stop_event.is_set():
            ret_0, frame_0 = self.cap_0.read()

            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps0 = frame_count / elapsed_time
                print(f"[CamCapture] 实时推理帧率: {fps0:.2f} FPS")
                frame_count = 0
                start_time = time.time()

            if ret_0:
                frame_count += 1
                frame_0 = cv2.resize(frame_0, (640, 640), interpolation=cv2.INTER_AREA)
                lane_infer_ret = self.run_inference(frame_0)
                lane_infer_ret[0][0] *= 1.1

                ret_bytes = struct.pack("=ff", lane_infer_ret[0][0], lane_infer_ret[0][1])
                self.shm_lane_ret_bytes.buf[0:len(ret_bytes)] = ret_bytes
                self.process_frame(
                    frame_0,
                    np.ndarray(frame_0.shape, dtype=frame_0.dtype, buffer=self.shm_image_0.buf),
                )

        print("[CamCapture] 收到终止信号，准备退出")
        if self.cap_0 is not None:
            self.cap_0.release()


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
