import time

import numpy as np

from box_pid import BoxPidAligner
from get_json import load_params
from shared_memory_manager import SharedMemoryManager
from tool_func import (
    DEFAULT_SEED_LABEL_ALIASES,
    DEFAULT_SEED_DROP_SLOT_MAP,
    get_biggest_box,
    get_seed_box,
    has_all_seed_targets,
    get_only_box,
    judge_seed_layout,
)


class Base_func:
    """Base control bridge for chassis and actuator commands."""

    def __init__(self, mode="normal", request_queue=None, base_file="motion.json", use_lane_shm=True):
        self.mode = mode
        self.request_queue = request_queue
        self.shm_manager = SharedMemoryManager()
        self.use_lane_shm = use_lane_shm
        if self.use_lane_shm:
            self.shm_manager.create_block("shm_lane", size=8)
        motion_cfg = load_params(filename=base_file)
        self.base_actions = motion_cfg.get("BASE_MOTION", {})
        self.arm_actions = motion_cfg.get("ARM_MOTION", {})

    def MOD_LANE(self, base_speed=-0.28):
        if not self.use_lane_shm:
            print("[MoveBase] Lane shared memory is disabled in this mode.")
            return
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

    def execute_arm_action(self, action_key, countdown=1.0):
        motion_list = self.arm_actions.get(action_key, [])
        if not motion_list:
            print(f"[MoveBase] Arm action not found: {action_key}")
            return False

        for motion in motion_list:
            self.send_motion_command(motion)
            time.sleep(0.02)
            time.sleep(countdown)
        return True

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
        task_cfg = load_params("motion.json").get("TASK_CONFIG", {})
        self.seeding_cfg = task_cfg.get("SEEDING", {})

    def task1_executor(self):
        return self.Hanoi_Tower()

    def Hanoi_Tower(self):
        print("[Task_func] Executing Task 1 seeding task...")
        if self.task_client is None:
            print("[Task_func] Task model client is not configured.")
            return False

        cfg = self.seeding_cfg
        label_aliases = cfg.get("label_aliases", DEFAULT_SEED_LABEL_ALIASES)
        drop_slot_map = cfg.get("drop_slot_map", DEFAULT_SEED_DROP_SLOT_MAP)
        sort_reverse = bool(cfg.get("sort_reverse", False))
        ready_timeout = float(cfg.get("ready_timeout_s", 6.0))
        ready_confirm_frames = int(cfg.get("ready_confirm_frames", 3))
        layout_timeout = float(cfg.get("layout_timeout_s", 6.0))
        layout_confirm_frames = int(cfg.get("layout_confirm_frames", 2))
        track_timeout = float(cfg.get("track_timeout_s", 5.0))
        max_missed_frames = int(cfg.get("max_missed_frames", 8))
        cam_pose = cfg.get("cam_pose", "L")
        search_action = cfg.get("search_action", "moveshort")
        search_countdown = float(cfg.get("search_countdown", 0.15))
        search_max_steps = int(cfg.get("search_max_steps", 8))
        pickup_targets = cfg.get("pickup_target_pose", {})
        return_action_name = cfg.get("return_action", "SeedReturnLane")

        ready_count = 0
        ready_start = time.time()
        search_steps = 0
        while time.time() - ready_start < ready_timeout:
            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            if has_all_seed_targets(detections, label_aliases=label_aliases):
                ready_count += 1
                if ready_count >= ready_confirm_frames:
                    break
            else:
                ready_count = 0
                if search_steps < search_max_steps:
                    self.base.execute_base_motion(search_action, countdown=search_countdown)
                    search_steps += 1
            time.sleep(0.05)

        if ready_count < ready_confirm_frames:
            print("[Task_func] Seed targets were not all ready within timeout.")
            return False

        layout = None
        layout_count = 0
        last_signature = None
        start_time = time.time()
        while time.time() - start_time < layout_timeout:
            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            layout = judge_seed_layout(
                detections,
                label_aliases=label_aliases,
                drop_slot_map=drop_slot_map,
                sort_reverse=sort_reverse,
            )
            if layout is None:
                layout_count = 0
                last_signature = None
            else:
                signature = layout["signature"]
                if signature == last_signature:
                    layout_count += 1
                else:
                    last_signature = signature
                    layout_count = 1
                if layout_count >= layout_confirm_frames:
                    break
            time.sleep(0.05)

        if layout is None or layout_count < layout_confirm_frames:
            print("[Task_func] Failed to build seeding layout within timeout.")
            return False

        print(f"[Task_func] Seeding layout: {layout['slot_map']}")

        seed_order = ("large", "medium", "small")
        for index, seed_key in enumerate(seed_order):
            pickup_pose = tuple(pickup_targets.get(seed_key, [320, 260]))
            if not self.tracking_executor(
                target_pose=pickup_pose,
                cam_pose=cam_pose,
                timeout_s=track_timeout,
                max_missed_frames=max_missed_frames,
                selector=get_seed_box,
                selector_kwargs={"size_key": seed_key, "label_aliases": label_aliases},
            ):
                print(f"[Task_func] Failed to align with {seed_key} seed.")
                return False

            self._seed_grab(seed_key)

            move_action = layout["transport_map"][seed_key]["go_action"]
            self.base.execute_base_motion(move_action, countdown=search_countdown)
            self._seed_place(seed_key, layout["transport_map"][seed_key]["drop_slot"])

            if index < len(seed_order) - 1:
                next_seed = seed_order[index + 1]
                current_drop_slot = layout["transport_map"][seed_key]["drop_slot"].upper()
                next_pick_slot = layout["transport_map"][next_seed]["pick_slot"]
                back_action = f"Seed{current_drop_slot}To{next_pick_slot}"
                self.base.execute_base_motion(back_action, countdown=search_countdown)

        self.base.execute_base_motion(return_action_name, countdown=search_countdown)
        self.base.MOD_STOP()
        print("[Task_func] Task 1 seeding flow finished.")
        return True

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
        selector=None,
        selector_kwargs=None,
    ):
        if self.task_client is None:
            print("[Task_func] Tracking client is not configured.")
            self.base.MOD_STOP()
            return False

        self.tracking_aligner.reset()
        start_time = time.time()
        missed_frames = 0
        selector = selector or get_only_box
        selector_kwargs = selector_kwargs or {}

        while True:
            if timeout_s is not None and time.time() - start_time > timeout_s:
                if callable(debug_hook):
                    debug_hook(status="timeout", target_box=None, debug=None, missed_frames=missed_frames)
                self.tracking_aligner.reset()
                self.base.MOD_STOP()
                print("[Tracking] Timeout")
                return False

            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            target_box = selector(detections, **selector_kwargs)
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

    def _seed_grab(self, seed_key):
        print(f"[Task_func] Grab {seed_key} cylinder.")
        time.sleep(0.2)
        return True

    def _seed_place(self, seed_key, drop_slot):
        print(f"[Task_func] Place {seed_key} cylinder to slot {drop_slot}.")
        time.sleep(0.2)
        return True
