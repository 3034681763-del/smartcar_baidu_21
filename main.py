import multiprocessing
import sys
import time

from Cam_cap import vidpub_course
from Process_manage import ProcessManager
from SerialCommunicate import serial_server_process
from State_Handle import TaskManager
from infer_server_client import model_configs, serve_model_process
from shared_memory_manager import SharedMemoryManager

SERIAL_PHYSICAL_PATH = "2.2:1.0"
LANE_CAMERA_DEVICE = "/dev/video0"
TASK_CAMERA_DEVICE = "/dev/video1"
LANE_MODEL_PATH = "/home/jetson/workspace_plus/vehicle_wbt_21th_lane/src/cnn_auto.nb"
ENABLE_AUX_MODELS = True


def start_framework():
    print("=" * 60)
    print(" SmartCar 21 Lane Framework With OCR Support ")
    print("=" * 60)

    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_0", size=640 * 640 * 3)
    shm_manager.create_block("shm_task", size=640 * 640 * 3)
    shm_manager.create_block("shm_lane", size=8)
    shm_manager.create_block("shm_crop", size=640 * 640 * 3)
    request_queue = multiprocessing.Queue()
    publish_queue = multiprocessing.Queue()

    mgr = ProcessManager()

    print("[Main] Starting UART service process...")
    mgr.add_process(
        target=serial_server_process,
        args=(SERIAL_PHYSICAL_PATH, 115200, request_queue, publish_queue),
    )

    print("[Main] Starting lane/task camera process...")
    mgr.add_process(
        target=vidpub_course,
        args=("shm_0", "shm_task", "shm_lane", LANE_MODEL_PATH, LANE_CAMERA_DEVICE, TASK_CAMERA_DEVICE),
    )

    if ENABLE_AUX_MODELS:
        print("[Main] Starting task and OCR model services...")
        for cfg in model_configs:
            mgr.add_process(target=serve_model_process, args=(cfg, shm_manager))
    else:
        print("[Main] Auxiliary task/OCR model services are disabled.")

    mgr.start_all()

    time.sleep(1.5)

    print("[Main] Starting state manager...")
    brain = TaskManager(
        request_queue=request_queue,
        publish_queue=publish_queue,
        enable_aux_models=ENABLE_AUX_MODELS,
        task_shm_key="shm_task",
    )
    brain.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, shutting down...")
        brain.cleanup()
        mgr.terminate_all()
        sys.exit(0)


if __name__ == "__main__":
    start_framework()
