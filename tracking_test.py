#!/usr/bin/env python3
import argparse
import multiprocessing
import time
from multiprocessing import shared_memory
from queue import Queue

import cv2
import numpy as np

from camera_device_utils import open_camera_from_device_arg
from Process_manage import ProcessManager
from SerialCommunicate import SerialServer
from infer_server_client import InferClient1, model_configs, serve_model_process
from move_base import Base_func, Task_func
from shared_memory_manager import SharedMemoryManager


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone tracking mode test driven only by the task camera."
    )
    parser.add_argument(
        "--task-device",
        default="2.1:1.0",
        help="Task camera device path, /dev/videoX, or physical path like 2.1:1.0",
    )
    parser.add_argument(
        "--physical-path",
        default="2.2:1.0",
        help="Serial physical path, for example 2.2:1.0 or /dev/ttyUSB0",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--target-x", type=float, default=320.0, help="Tracking target x")
    parser.add_argument("--target-y", type=float, default=240.0, help="Tracking target y")
    parser.add_argument("--cam-pose", default="L", choices=["L", "R", "shoot"], help="Tracking pose")
    parser.add_argument("--timeout", type=float, default=5.0, help="Tracking timeout")
    return parser


def publish_task_camera_only(stop_event, task_shm_key, task_device, width=640, height=480, fps=60, codec="MJPG"):
    task_cap = None
    task_shm = None
    window_name = "Task Preview"
    try:
        task_cap, actual_device, candidates = open_camera_from_device_arg(
            task_device,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            role_name="Task",
        )
        print(f"[tracking_test] Task path:  {task_device}")
        print(f"[tracking_test] Task nodes: {candidates}")
        print(f"[tracking_test] Task using: {actual_device}")

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
                print("[tracking_test] q pressed, stopping task camera publisher")
                stop_event.set()
                break
    finally:
        if task_cap is not None:
            task_cap.release()
        if task_shm is not None:
            task_shm.close()
        cv2.destroyAllWindows()


def main():
    args = build_parser().parse_args()
    multiprocessing.freeze_support()

    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_task", size=640 * 640 * 3)
    shm_manager.create_block("shm_crop", size=640 * 640 * 3)

    request_queue = Queue()
    publish_queue = Queue()

    serial_server = SerialServer(
        serial_path=args.physical_path,
        baudrate=args.baudrate,
        request_queue=request_queue,
        publish_queue=publish_queue,
    )
    serial_server.run()

    manager = ProcessManager()
    manager.add_process(
        target=publish_task_camera_only,
        args=(
            "shm_task",
            args.task_device,
        ),
    )
    for cfg in model_configs:
        manager.add_process(target=serve_model_process, args=(cfg, shm_manager))
    manager.start_all()

    task_client = None
    try:
        task_cfg = next(cfg for cfg in model_configs if cfg["name"] == "task")
        task_client = InferClient1("task", shm_manager, task_cfg["port"])
        base = Base_func(request_queue=request_queue)
        task = Task_func(base, ocr_reader=None, task_shm_key="shm_task", task_client=task_client)

        print("=" * 60)
        print("Tracking Test")
        print(f"Task camera: {args.task_device}")
        print(f"Target pose: ({args.target_x}, {args.target_y})")
        print(f"cam_pose:    {args.cam_pose}")
        print(f"timeout:     {args.timeout:.2f}s")
        print("=" * 60)
        time.sleep(1.0)

        result = task.tracking_executor(
            target_pose=(args.target_x, args.target_y),
            cam_pose=args.cam_pose,
            timeout_s=args.timeout,
        )
        print(f"[tracking_test] result={result}")
        return 0 if result else 1
    except KeyboardInterrupt:
        print("[tracking_test] Stopped by user.")
        return 0
    finally:
        if task_client is not None:
            task_client.close()
        serial_server.close()
        manager.terminate_all()
        shm_manager.release_all()


if __name__ == "__main__":
    raise SystemExit(main())
