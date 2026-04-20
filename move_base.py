import time

import numpy as np

from box_pid import BoxPidAligner
from get_json import load_params
from shared_memory_manager import SharedMemoryManager
from tool_func import get_only_box


class Base_func:
    """Base control bridge for chassis and actuator commands."""

    def __init__(self, mode="normal", request_queue=None, base_file="motion.json"):
        self.mode = mode
        self.request_queue = request_queue
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_block("shm_lane", size=8)
        self.base_actions = load_params(filename=base_file).get("BASE_MOTION", {})

    def MOD_LANE(self, base_speed=-0.28):
        try:
            angle, infer_speed = self.shm_manager.read_floats("shm_lane", 2)
            del infer_speed

            if angle > 120.0:
                angle = 120.0
            if angle < -120.0:
                angle = -120.0

            target_speed = base_speed
            print(
                f"[MoveBase] MOD_LANE -> Executing Deviation: {angle:.2f}, "
                f"Speed: {target_speed:.2f}"
            )

            data = {
                "cmd": "Motion",
                "mode": 0,
                "deviation": float(angle),
                "speed_x": float(target_speed),
            }
            if self.request_queue is None:
                print("[MoveBase] Request queue is not configured.")
                return
            self.request_queue.put(data)
        except Exception as exc:
            print(f"[MoveBase] Lane Following Exec Error: {exc}")

    def MOD_STOP(self):
        data = {"cmd": "Motion", "mode": 1, "pos_x": 0, "pos_y": 0, "z_angle": 0}
        if self.request_queue is not None:
            self.request_queue.put(data)

    def send_motion_command(self, data):
        if self.request_queue is not None:
            self.request_queue.put(data)

    def execute_base_motion(self, action_key, countdown=1.0):
        motion_list = self.base_actions.get(action_key, [])
        if not motion_list:
            print(f"[MoveBase] Base action not found: {action_key}")
            return False

        for motion in motion_list:
            self.send_motion_command(motion)
            time.sleep(0.02)
            time.sleep(countdown)
        return True

    def execute_arm_motion(self, mot0, mot1, mot2, mot3, suck=0, light=0):
        data = {
            "cmd": "Arm",
            "mot0": mot0,
            "mot1": mot1,
            "mot2": mot2,
            "mot3": mot3,
            "suck": suck,
            "light": light,
        }
        if self.request_queue is not None:
            self.request_queue.put(data)
        return {"status": "queued", "cmd": "Arm"}

    def set_sys_mode(self, flag):
        data = {"cmd": "SysMode", "flag": flag}
        if self.request_queue is not None:
            self.request_queue.put(data)
        return {"status": "queued", "cmd": "SysMode"}


class Task_func:
    """Placeholder task library. Task2 now demonstrates the OCR call chain."""

    def __init__(
        self,
        base_func: Base_func,
        ocr_reader=None,
        task_shm_key="shm_task",
        task_client=None,
    ):
        self.base = base_func
        self.ocr_reader = ocr_reader
        self.task_shm_key = task_shm_key
        self.task_client = task_client
        self.tracking_aligner = BoxPidAligner(params="Fast")

    def task1_executor(self):
        print("[Task_func] Executing Task 1 placeholder...")

    def task2_executor(self):
        print("[Task_func] Executing Task 2 OCR placeholder...")
        if self.ocr_reader is None:
            print("[Task_func] OCR reader is not configured.")
            return

        texts = self.ocr_reader.read_texts(
            shm_key=self.task_shm_key,
            shape=(640, 640, 3),
            result_amount=1,
            sort_by="y",
        )
        if texts:
            print(f"[Task_func] OCR result: {texts}")
        else:
            print("[Task_func] OCR returned no text.")

    def task3_executor(self):
        print("[Task_func] Executing Task 3 placeholder...")

    def task4_executor(self):
        print("[Task_func] Executing Task 4 placeholder...")

    def base_motion_test_executor(self):
        print("[Task_func] Executing base motion group test...")
        test_sequence = [
            ("Default", 0.3),
            ("moveshort", 0.6),
            ("backshort", 0.6),
            ("movelong", 0.8),
            ("backlong", 0.8),
            ("TurnLeftSmall", 0.6),
            ("TurnRightSmall", 0.6),
            ("TurnLeft", 0.8),
            ("TurnRight", 0.8),
            ("Default", 0.3),
        ]

        for action_key, countdown in test_sequence:
            print(f"[Task_func] Base motion -> {action_key}")
            if not self.base.execute_base_motion(action_key, countdown=countdown):
                print(f"[Task_func] Base motion test aborted at: {action_key}")
                return False

        print("[Task_func] Base motion group test finished.")
        return True

    def tracking_executor(
        self,
        target_pose=(320, 240),
        cam_pose="L",
        timeout_s=5.0,
        max_missed_frames=5,
        debug_hook=None,
    ):
        if self.task_client is None:
            print("[Task_func] Tracking client is not configured.")
            self.base.MOD_STOP()
            return False

        self.tracking_aligner.reset()
        start_time = time.time()
        missed_frames = 0

        while True:
            if timeout_s is not None and time.time() - start_time > timeout_s:
                if callable(debug_hook):
                    debug_hook(status="timeout", target_box=None, debug=None, missed_frames=missed_frames)
                self.tracking_aligner.reset()
                self.base.MOD_STOP()
                print("[Tracking] Timeout")
                return False

            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            target_box = get_only_box(detections)
            if target_box is None or getattr(target_box, "center", None) is None:
                missed_frames += 1
                if callable(debug_hook):
                    debug_hook(
                        status="missing",
                        target_box=None,
                        debug=None,
                        missed_frames=missed_frames,
                    )
                if missed_frames >= max_missed_frames:
                    self.tracking_aligner.reset()
                    self.base.MOD_STOP()
                    print(f"[Tracking] No target box for {missed_frames} frames.")
                    return False
                time.sleep(0.02)
                continue

            missed_frames = 0

            aligned, motion, debug = self.tracking_aligner.update(
                target_box.center,
                target_pose=target_pose,
                cam_pose=cam_pose,
            )
            if callable(debug_hook):
                debug_hook(
                    status="tracking",
                    target_box=target_box,
                    debug=debug,
                    missed_frames=missed_frames,
                )
            self.base.send_motion_command(motion)

            x_err = debug["x_err"]
            y_err = debug["y_err"]
            x_text = "skip" if x_err is None else f"{x_err:.2f}"
            y_text = "skip" if y_err is None else f"{y_err:.2f}"

            if aligned:
                if callable(debug_hook):
                    debug_hook(
                        status="aligned",
                        target_box=target_box,
                        debug=debug,
                        missed_frames=missed_frames,
                    )
                self.base.MOD_STOP()
                self.tracking_aligner.reset()
                print("[Tracking] Location OK")
                return True

            print(
                "[Tracking] "
                f"center={target_box.center} "
                f"x_err={x_text} y_err={y_text}"
            )
            time.sleep(0.02)
