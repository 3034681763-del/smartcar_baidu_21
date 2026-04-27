import time

import numpy as np

from box_pid import BoxPidAligner
from get_json import load_params
from irrigation_task import IrrigationTaskExecutor
from order_delivery_task import OrderDeliveryTaskExecutor
from pest_confirm_task import PestConfirmTaskExecutor
from place_delivery_task import PlaceDeliveryTaskExecutor
from shooting_task import ShootingTaskExecutor
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

    def __init__(
        self,
        mode="normal",
        request_queue=None,
        base_file="motion.json",
        use_lane_shm=True,
        action_done_event=None,
    ):
        self.mode = mode
        self.request_queue = request_queue
        self.action_done_event = action_done_event
        self.shm_manager = SharedMemoryManager()
        self.use_lane_shm = use_lane_shm
        if self.use_lane_shm:
            self.shm_manager.create_block("shm_lane", size=8)
        motion_cfg = load_params(filename=base_file)
        self.base_actions = motion_cfg.get("BASE_MOTION", {})
        self.arm_actions = motion_cfg.get("ARM_MOTION", {})
        action_ack_cfg = motion_cfg.get("ACTION_ACK", {})
        self.default_action_ack_timeout_s = float(action_ack_cfg.get("wait_timeout_s", 8.0))

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
            if self.action_done_event is None:
                print(f"[MoveBase] Action-done event is not configured, cannot run: {action_key}")
                return False
            wait_timeout_s = float(motion.get("wait_timeout_s", self.default_action_ack_timeout_s))
            self.action_done_event.clear()
            self.send_motion_command(motion)
            if not self.action_done_event.wait(timeout=wait_timeout_s):
                print(f"[MoveBase] Action wait timeout: {action_key} ({wait_timeout_s:.2f}s)")
                self.MOD_STOP()
                return False
            self.action_done_event.clear()
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
            if self.action_done_event is None:
                print(f"[MoveBase] Action-done event is not configured, cannot run arm action: {action_key}")
                return False
            wait_timeout_s = float(motion.get("wait_timeout_s", self.default_action_ack_timeout_s))
            self.action_done_event.clear()
            self.send_motion_command(motion)
            if not self.action_done_event.wait(timeout=wait_timeout_s):
                print(f"[MoveBase] Arm action wait timeout: {action_key} ({wait_timeout_s:.2f}s)")
                self.MOD_STOP()
                return False
            self.action_done_event.clear()
        return True

    def set_sys_mode(self, flag):
        data = {"cmd": "SysMode", "flag": flag}
        if self.request_queue is not None:
            self.request_queue.put(data)
        return {"status": "queued", "cmd": "SysMode"}

    def execute_shoot_instruction(self, instruction=None, wait_timeout_s=None):
        return self.execute_wait_instruction(
            instruction or {"cmd": "SysMode", "flag": 9},
            wait_timeout_s=wait_timeout_s,
            action_name="shoot",
        )

    def execute_wait_instruction(self, instruction, wait_timeout_s=None, action_name="custom action"):
        if self.request_queue is None:
            print(f"[MoveBase] Request queue is not configured, cannot run {action_name}.")
            return False

        timeout_s = self.default_action_ack_timeout_s if wait_timeout_s is None else float(wait_timeout_s)
        if self.action_done_event is None:
            print(f"[MoveBase] Action-done event is not configured, cannot wait for {action_name} ack.")
            return False

        self.action_done_event.clear()
        self.request_queue.put(dict(instruction))
        if not self.action_done_event.wait(timeout=timeout_s):
            print(f"[MoveBase] {action_name} wait timeout ({timeout_s:.2f}s)")
            self.MOD_STOP()
            return False
        self.action_done_event.clear()
        return True

    def execute_chassis_instruction(self, instruction, countdown=0.3):
        if not instruction:
            return True
        print(f"[MoveBase] Chassis instruction -> {instruction}")
        if instruction in self.base_actions:
            return self.execute_base_motion(instruction, countdown=countdown)
        time.sleep(max(countdown, 0.05))
        return True

    def execute_arm_instruction(self, instruction, countdown=0.2):
        if not instruction:
            return True
        print(f"[MoveBase] Arm instruction -> {instruction}")
        if instruction in self.arm_actions:
            return self.execute_arm_action(instruction, countdown=countdown)
        time.sleep(max(countdown, 0.05))
        return True


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
        self.task_shm = SharedMemoryManager()
        self.task_shm.create_block(task_shm_key, size=640 * 640 * 3)
        task_cfg = load_params("motion.json").get("TASK_CONFIG", {})
        self.seeding_cfg = task_cfg.get("SEEDING", {})
        self.irrigation_cfg = task_cfg.get("IRRIGATION", {})
        self.irrigation_executor_impl = IrrigationTaskExecutor(
            self.base,
            ocr_reader=self.ocr_reader,
            task_client=self.task_client,
            task_shm_key=self.task_shm_key,
            tracking_callback=self.tracking_executor,
        )
        self.pest_confirm_executor_impl = PestConfirmTaskExecutor(
            self.base,
            task_client=self.task_client,
            task_shm=self.task_shm,
            task_shm_key=self.task_shm_key,
            tracking_callback=self.tracking_executor,
        )
        self.shooting_executor_impl = ShootingTaskExecutor(
            self.base,
            task_client=self.task_client,
            task_shm_key=self.task_shm_key,
            tracking_callback=self.tracking_executor,
        )
        self.order_delivery_executor_impl = OrderDeliveryTaskExecutor(
            self.base,
            ocr_reader=self.ocr_reader,
            task_client=self.task_client,
            task_shm_key=self.task_shm_key,
            tracking_callback=self.tracking_executor,
        )
        self.place_delivery_executor_impl = PlaceDeliveryTaskExecutor(
            self.base,
            ocr_reader=self.ocr_reader,
            task_client=self.task_client,
            task_shm_key=self.task_shm_key,
            tracking_callback=self.tracking_executor,
        )

    def task1_executor(self):
        return self.seeding_executor()

    def seeding_executor(self):
        return self.Hanoi_Tower()

    def Hanoi_Tower(self):
        print("[Task_func] Executing Seeding task...")
        if self.task_client is None:
            print("[Task_func] Task model client is not configured.")
            return False

        cfg = self.seeding_cfg
        label_aliases = cfg.get("label_aliases", DEFAULT_SEED_LABEL_ALIASES)
        drop_slot_map = cfg.get("drop_slot_map", DEFAULT_SEED_DROP_SLOT_MAP)
        sort_reverse = bool(cfg.get("sort_reverse", False))
        ready_timeout = float(cfg.get("ready_timeout_s", 6.0))
        ready_confirm_frames = int(cfg.get("ready_confirm_frames", 3))
        ready_min_area = float(cfg.get("ready_min_area", 0.0))
        ready_edge_margin = int(cfg.get("ready_edge_margin", 5))
        ready_frame_width = int(cfg.get("ready_frame_width", 640))
        ready_frame_height = int(cfg.get("ready_frame_height", 640))
        layout_timeout = float(cfg.get("layout_timeout_s", 6.0))
        layout_confirm_frames = int(cfg.get("layout_confirm_frames", 2))
        layout_min_score = float(cfg.get("layout_min_score", 0.5))
        use_color_assist = bool(cfg.get("use_color_assist", True))
        color_patch_radius = int(cfg.get("color_patch_radius", 12))
        track_timeout = float(cfg.get("track_timeout_s", 5.0))
        max_missed_frames = int(cfg.get("max_missed_frames", 8))
        recover_pause_s = float(cfg.get("tracking_recover_pause_s", 0.06))
        recover_timeout_s = float(cfg.get("tracking_recover_timeout_s", 0.5))
        tracking_retry_count = int(cfg.get("tracking_retry_count", 2))
        seed_phase_timeout = float(cfg.get("seed_phase_timeout_s", 12.0))
        cam_pose = cfg.get("cam_pose", "L")
        search_actions = list(cfg.get("search_actions", []))
        if not search_actions:
            fallback_search_action = cfg.get("search_action", "moveshort")
            search_actions = [fallback_search_action]
        search_countdown = float(cfg.get("search_countdown", 0.15))
        search_max_steps = int(cfg.get("search_max_steps", 8))
        pickup_targets = cfg.get("pickup_target_pose", {})
        return_action_name = cfg.get("return_action", "SeedReturnLane")
        abort_action_name = cfg.get("abort_action", None)
        abort_countdown = float(cfg.get("abort_countdown", search_countdown))

        ready_count = 0
        ready_start = time.time()
        search_steps = 0
        search_index = 0
        while time.time() - ready_start < ready_timeout:
            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            ready_state = has_all_seed_targets(
                detections,
                label_aliases=label_aliases,
                min_area=ready_min_area,
                edge_margin=ready_edge_margin,
                frame_width=ready_frame_width,
                frame_height=ready_frame_height,
                return_details=True,
            )
            if ready_state.get("ready", False):
                ready_count += 1
                print(f"[Task_func] Seed ready stable: {ready_count}/{ready_confirm_frames}")
                if ready_count >= ready_confirm_frames:
                    break
            else:
                ready_count = 0
                missing_reason = self._describe_seed_ready_issue(ready_state)
                print(f"[Task_func] Seed ready wait: {missing_reason}")
                if search_steps < search_max_steps:
                    search_index = self._run_search_action(search_actions, search_index, search_countdown)
                    search_steps += 1
            time.sleep(0.05)

        if ready_count < ready_confirm_frames:
            return self._task_fail(
                "Seed targets were not all ready within timeout.",
                abort_action_name,
                abort_countdown,
            )

        layout = None
        layout_count = 0
        last_signature = None
        start_time = time.time()
        search_steps = 0
        while time.time() - start_time < layout_timeout:
            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            frame = self._read_task_frame()
            layout = judge_seed_layout(
                detections,
                image=frame,
                label_aliases=label_aliases,
                drop_slot_map=drop_slot_map,
                sort_reverse=sort_reverse,
                min_score=layout_min_score,
                use_color_assist=use_color_assist,
                color_patch_radius=color_patch_radius,
            )
            if layout is None:
                layout_count = 0
                last_signature = None
                if search_steps < search_max_steps:
                    search_index = self._run_search_action(search_actions, search_index, search_countdown)
                    search_steps += 1
            else:
                signature = layout["signature"]
                if signature == last_signature:
                    layout_count += 1
                else:
                    last_signature = signature
                    layout_count = 1
                print(
                    f"[Task_func] Layout candidate {layout['slot_map']} "
                    f"({layout['source']}), stable {layout_count}/{layout_confirm_frames}"
                )
                if layout_count >= layout_confirm_frames:
                    break
            time.sleep(0.05)

        if layout is None or layout_count < layout_confirm_frames:
            return self._task_fail(
                "Failed to build seeding layout within timeout.",
                abort_action_name,
                abort_countdown,
            )

        print(f"[Task_func] Seeding layout: {layout['slot_map']}")

        seed_order = ("large", "medium", "small")
        for index, seed_key in enumerate(seed_order):
            seed_phase_start = time.time()
            pickup_pose = tuple(pickup_targets.get(seed_key, [320, 260]))
            aligned = False
            align_attempt = 0
            while align_attempt <= tracking_retry_count and time.time() - seed_phase_start < seed_phase_timeout:
                print(
                    f"[Task_func] Align {seed_key} seed, "
                    f"attempt {align_attempt + 1}/{tracking_retry_count + 1}"
                )
                aligned = self.tracking_executor(
                    target_pose=pickup_pose,
                    cam_pose=cam_pose,
                    timeout_s=track_timeout,
                    max_missed_frames=max_missed_frames,
                    recover_pause_s=recover_pause_s,
                    recover_timeout_s=recover_timeout_s,
                    selector=get_seed_box,
                    selector_kwargs={"size_key": seed_key, "label_aliases": label_aliases},
                )
                if aligned:
                    break
                align_attempt += 1
                if align_attempt <= tracking_retry_count:
                    search_index = self._run_search_action(search_actions, search_index, search_countdown)

            if not aligned:
                return self._task_fail(
                    f"Failed to align with {seed_key} seed.",
                    abort_action_name,
                    abort_countdown,
                )

            self._seed_grab(seed_key)

            move_action = layout["transport_map"][seed_key]["go_action"]
            if not self.base.execute_base_motion(move_action, countdown=search_countdown):
                return self._task_fail(
                    f"Base action missing: {move_action}",
                    abort_action_name,
                    abort_countdown,
                )
            self._seed_place(seed_key, layout["transport_map"][seed_key]["drop_slot"])

            if index < len(seed_order) - 1:
                next_seed = seed_order[index + 1]
                current_drop_slot = layout["transport_map"][seed_key]["drop_slot"].upper()
                next_pick_slot = layout["transport_map"][next_seed]["pick_slot"]
                back_action = f"Seed{current_drop_slot}To{next_pick_slot}"
                if not self.base.execute_base_motion(back_action, countdown=search_countdown):
                    return self._task_fail(
                        f"Base action missing: {back_action}",
                        abort_action_name,
                        abort_countdown,
                    )

            if time.time() - seed_phase_start >= seed_phase_timeout:
                return self._task_fail(
                    f"Seed phase timeout for {seed_key}.",
                    abort_action_name,
                    abort_countdown,
                )

        if not self.base.execute_base_motion(return_action_name, countdown=search_countdown):
            return self._task_fail(
                f"Base action missing: {return_action_name}",
                abort_action_name,
                abort_countdown,
            )
        self.base.MOD_STOP()
        print("[Task_func] Seeding task flow finished.")
        return True

    def task2_executor(self):
        return self.irrigation_executor()

    def pest_confirm_executor(self):
        return self.pest_confirm_executor_impl.run()

    def shooting_executor(self, pest_results):
        return self.shooting_executor_impl.run(pest_results)

    def order_delivery_executor(self):
        return self.order_delivery_executor_impl.run()

    def get_last_order(self):
        return self.order_delivery_executor_impl.last_order

    def place_delivery_executor(self, order):
        return self.place_delivery_executor_impl.run(order)

    def has_order_machine(self):
        if self.task_client is None:
            return False
        return self.order_delivery_executor_impl.has_order_machine()

    def has_unit(self, building="1"):
        if self.task_client is None:
            return False
        return self.place_delivery_executor_impl.has_unit(building)

    def has_pest_animal(self):
        if self.task_client is None:
            return False
        return self.pest_confirm_executor_impl.has_animal()

    def task3_executor(self):
        print("[Task_func] Executing Task 3 placeholder...")

    def task4_executor(self):
        print("[Task_func] Executing Task 4 placeholder...")

    def irrigation_executor(self, supply_counts=None):
        return self.irrigation_executor_impl.run(supply_counts=supply_counts)

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
        recover_pause_s=0.05,
        recover_timeout_s=0.5,
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
        recover_start = None
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
                if recover_start is None:
                    recover_start = time.time()
                    self.base.MOD_STOP()
                if callable(debug_hook):
                    debug_hook(
                        status="missing",
                        target_box=None,
                        debug=None,
                        missed_frames=missed_frames,
                    )
                if recover_timeout_s is not None and recover_start is not None:
                    if time.time() - recover_start > recover_timeout_s:
                        self.tracking_aligner.reset()
                        self.base.MOD_STOP()
                        print(f"[Tracking] Recovery timeout after {missed_frames} missed frames.")
                        return False
                if missed_frames >= max_missed_frames:
                    self.tracking_aligner.reset()
                    self.base.MOD_STOP()
                    print(f"[Tracking] No target box for {missed_frames} frames.")
                    return False
                time.sleep(recover_pause_s)
                continue

            missed_frames = 0
            recover_start = None

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

    def _read_task_frame(self):
        return self.task_shm.read_image(self.task_shm_key, (640, 640, 3), np.uint8)

    def _run_search_action(self, search_actions, search_index, countdown):
        if not search_actions:
            return search_index
        action = search_actions[search_index % len(search_actions)]
        print(f"[Task_func] Search action -> {action}")
        self.base.execute_base_motion(action, countdown=countdown)
        return search_index + 1

    def _task_fail(self, reason, abort_action_name=None, abort_countdown=0.15):
        self.base.MOD_STOP()
        if abort_action_name:
            self.base.execute_base_motion(abort_action_name, countdown=abort_countdown)
        print(f"[Task_func] Seeding task failed: {reason}")
        return False

    def _describe_seed_ready_issue(self, ready_state):
        issues = []
        for seed_key in ("large", "medium", "small"):
            state = ready_state.get(seed_key, {})
            if state.get("ready"):
                continue
            issues.append(f"{seed_key}:{state.get('reason', 'unknown')}")
        return ", ".join(issues) if issues else "unknown"

    def _seed_grab(self, seed_key):
        print(f"[Task_func] Grab {seed_key} cylinder.")
        time.sleep(0.2)
        return True

    def _seed_place(self, seed_key, drop_slot):
        print(f"[Task_func] Place {seed_key} cylinder to slot {drop_slot}.")
        time.sleep(0.2)
        return True
