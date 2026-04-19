#!/usr/bin/env python3
import argparse
import multiprocessing
import time
from queue import Queue

import numpy as np

from Cam_cap import vidpub_course
from Process_manage import ProcessManager
from SerialCommunicate import SerialServer
from infer_server_client import InferClient1, model_configs, serve_model_process
from move_base import Base_func, Task_func
from shared_memory_manager import SharedMemoryManager


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone tracking mode test."
    )
    parser.add_argument("--lane-device", default="/dev/video0", help="Lane camera device")
    parser.add_argument("--task-device", default="/dev/video1", help="Task camera device")
    parser.add_argument(
        "--lane-model",
        default="/home/jetson/workspace_plus/vehicle_wbt_21th_lane/src/cnn_auto.nb",
        help="Lane model path used by Cam_cap",
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


def main():
    args = build_parser().parse_args()
    multiprocessing.freeze_support()

    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_0", size=640 * 640 * 3)
    shm_manager.create_block("shm_task", size=640 * 640 * 3)
    shm_manager.create_block("shm_lane", size=8)
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
        target=vidpub_course,
        args=(
            "shm_0",
            "shm_task",
            "shm_lane",
            args.lane_model,
            args.lane_device,
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
        print(f"Lane camera: {args.lane_device}")
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
