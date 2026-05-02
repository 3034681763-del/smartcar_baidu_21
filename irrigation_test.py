#!/usr/bin/env python3
import argparse
import atexit
import json
import multiprocessing
import os
import tempfile
import time
from multiprocessing import shared_memory
from queue import Queue

from test_shutdown import ShutdownController

class FakeOCRReader:
    def __init__(self, tower_needs):
        self._tower_needs = list(tower_needs)
        self._index = 0

    def read_texts(self, **kwargs):
        if self._index >= len(self._tower_needs):
            value = self._tower_needs[-1]
        else:
            value = self._tower_needs[self._index]
            self._index += 1
        return [str(value)]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone irrigation task test with optional vision and serial."
    )
    parser.add_argument(
        "--physical-path",
        default="2.4:1.0",
        help="Serial physical path, for example 2.2:1.0 or /dev/ttyUSB0",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Run without opening the serial server. Useful for dry-run state testing.",
    )
    parser.add_argument(
        "--task-device",
        default="1-2.1:1.0",
        help="Task camera device path, /dev/videoX, or physical path like 1-2.1:1.0",
    )
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="Enable task camera and task model for real visual alignment.",
    )
    parser.add_argument(
        "--tower1-need",
        type=int,
        default=2,
        help="Fallback/forced demand for tower 1 in dry-run mode.",
    )
    parser.add_argument(
        "--tower2-need",
        type=int,
        default=1,
        help="Fallback/forced demand for tower 2 in dry-run mode.",
    )
    parser.add_argument("--supply-a", type=int, default=2, help="Initial block count in supply zone a.")
    parser.add_argument("--supply-b", type=int, default=2, help="Initial block count in supply zone b.")
    parser.add_argument("--supply-c", type=int, default=2, help="Initial block count in supply zone c.")
    parser.add_argument(
        "--countdown",
        type=float,
        default=0.15,
        help="Default action wait time for the generated irrigation config.",
    )
    parser.add_argument(
        "--arm-countdown",
        type=float,
        default=0.1,
        help="Default arm action wait time for the generated irrigation config.",
    )
    return parser


def publish_task_camera_only(
    stop_event,
    task_shm_key,
    task_device,
    width=640,
    height=480,
    fps=60,
    codec="MJPG",
):
    import cv2
    import numpy as np
    from camera_device_utils import open_camera_from_device_arg

    task_cap = None
    task_shm = None
    window_name = "Irrigation Task Preview"
    try:
        task_cap, actual_device, candidates = open_camera_from_device_arg(
            task_device,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            role_name="Task",
        )
        print(f"[irrigation_test] Task path:  {task_device}")
        print(f"[irrigation_test] Task nodes: {candidates}")
        print(f"[irrigation_test] Task using: {actual_device}")

        task_shm = shared_memory.SharedMemory(name=task_shm_key)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while not stop_event.is_set():
            ok, frame = task_cap.read()
            if not ok:
                continue

            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=task_shm.buf)
            shm_array[:] = frame[:]

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[irrigation_test] q pressed, stopping task camera publisher")
                stop_event.set()
                break
    finally:
        if task_cap is not None:
            task_cap.release()
        if task_shm is not None:
            task_shm.close()
        cv2.destroyAllWindows()


def build_temp_motion_file(args):
    with open("motion.json", "r", encoding="utf-8") as f:
        motion_cfg = json.load(f)

    irrigation_cfg = motion_cfg.setdefault("TASK_CONFIG", {}).setdefault("IRRIGATION", {})
    irrigation_cfg["supply_counts"] = {
        "a": int(args.supply_a),
        "b": int(args.supply_b),
        "c": int(args.supply_c),
    }
    irrigation_cfg["use_visual_alignment"] = bool(args.enable_vision)
    irrigation_cfg["default_need_count"] = int(args.tower1_need)
    irrigation_cfg["action_countdown"] = float(args.countdown)
    irrigation_cfg["arm_action_countdown"] = float(args.arm_countdown)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_irrigation_motion.json",
        prefix="smartcar_",
        delete=False,
        encoding="utf-8",
        dir=os.getcwd(),
    )
    with handle:
        json.dump(motion_cfg, handle, ensure_ascii=False, indent=2)

    atexit.register(lambda: os.path.exists(handle.name) and os.remove(handle.name))
    return handle.name


def main():
    shutdown = ShutdownController("irrigation_test").install()
    args = build_parser().parse_args()
    multiprocessing.freeze_support()
    from move_base import Base_func, Task_func
    from shared_memory_manager import SharedMemoryManager

    request_queue = Queue()
    publish_queue = Queue()
    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_task", size=640 * 640 * 3)

    serial_server = None
    if not args.no_serial:
        from SerialCommunicate import SerialServer

        serial_server = SerialServer(
            serial_path=args.physical_path,
            baudrate=args.baudrate,
            request_queue=request_queue,
            publish_queue=publish_queue,
        )
        serial_server.run()

    manager = None
    task_client = None
    ocr_reader = None
    motion_file = build_temp_motion_file(args)

    try:
        if args.enable_vision:
            from Process_manage import ProcessManager
            from infer_server_client import InferClient1, model_configs, serve_model_process

            manager = ProcessManager(setup_signal_handlers=False)
            manager.add_process(
                target=publish_task_camera_only,
                args=("shm_task", args.task_device),
            )
            for cfg in model_configs:
                manager.add_process(target=serve_model_process, args=(cfg, shm_manager))
            manager.start_all()
            time.sleep(1.0)

            task_cfg = next(cfg for cfg in model_configs if cfg["name"] == "task")
            task_client = InferClient1("task", shm_manager, task_cfg["port"])
        else:
            ocr_reader = FakeOCRReader([args.tower1_need, args.tower2_need])

        base = Base_func(request_queue=request_queue, base_file=motion_file, use_lane_shm=False)
        task = Task_func(base, ocr_reader=ocr_reader, task_shm_key="shm_task", task_client=task_client)
        task.irrigation_executor_impl.motion_file = motion_file

        if not args.enable_vision:
            task.irrigation_executor_impl.ocr_reader = ocr_reader
            task.irrigation_executor_impl.task_client = None
            task.irrigation_executor_impl.tracking_callback = None

        supply_counts = {"a": args.supply_a, "b": args.supply_b, "c": args.supply_c}

        print("=" * 60)
        print("Irrigation Task Test")
        print(f"Serial path:   {args.physical_path}")
        print(f"Serial mode:   {'disabled' if args.no_serial else 'enabled'}")
        print(f"Vision mode:   {args.enable_vision}")
        print(f"Tower needs:   {args.tower1_need}, {args.tower2_need}")
        print(f"Supply counts: a={args.supply_a}, b={args.supply_b}, c={args.supply_c}")
        if args.enable_vision:
            print(f"Task camera:   {args.task_device}")
        else:
            print("Task camera:   disabled (dry-run mode)")
        print("=" * 60)

        if shutdown.is_set():
            return 0
        result = task.irrigation_executor(supply_counts=supply_counts)
        runtime = task.irrigation_executor_impl.runtime
        print(f"[irrigation_test] result={result}")
        if runtime is not None:
            print(f"[irrigation_test] tower_need={runtime.tower_need}")
            print(f"[irrigation_test] tower_done={runtime.tower_done}")
            print(f"[irrigation_test] supply_counts={runtime.supply_counts}")
            print(f"[irrigation_test] completed={runtime.completed}")
        return 0 if result else 1
    except KeyboardInterrupt:
        shutdown.request()
        return 0
    finally:
        base = None
        try:
            request_queue.put({"cmd": "Motion", "mode": 1, "pos_x": 0, "pos_y": 0, "z_angle": 0})
            time.sleep(0.1)
        except Exception:
            pass
        if task_client is not None:
            task_client.close()
        if serial_server is not None:
            serial_server.close()
        if manager is not None:
            manager.terminate_all()
        shm_manager.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
