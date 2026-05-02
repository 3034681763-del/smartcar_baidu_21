import multiprocessing
import signal
import sys
import threading
import time

from Cam_cap import vidpub_course
from Process_manage import ProcessManager
from SerialCommunicate import serial_server_process
from State_Handle import TaskManager
from infer_server_client import model_configs, serve_model_process
from model_path_config import get_lane_model_path, get_model_profile
from shared_memory_manager import SharedMemoryManager

SERIAL_PHYSICAL_PATH = "2.4:1.0"
LANE_CAMERA_DEVICE = "1-2.3:1.0"
TASK_CAMERA_DEVICE = "1-2.1:1.0"
LANE_MODEL_PATH = get_lane_model_path()
ENABLE_AUX_MODELS = True


def start_framework():
    shutdown_event = threading.Event()
    shutdown_started = threading.Event()
    brain = None
    mgr = None
    shm_manager = None
    action_done_event = None

    def request_shutdown(signum=None, frame=None):
        del frame
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        if signum is None:
            print("\n[Main] Shutdown requested, stopping framework...")
        else:
            print(f"\n[Main] Signal {signum} received, stopping framework...")
        shutdown_event.set()
        if action_done_event is not None:
            action_done_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    print("=" * 60)
    print(" SmartCar 21 Lane Framework With OCR Support ")
    print("=" * 60)
    print(f"[Main] Model profile: {get_model_profile()}")
    print(f"[Main] Lane model path: {LANE_MODEL_PATH}")
    task_cfg = next((cfg for cfg in model_configs if cfg.get("name") == "task"), None)
    if task_cfg is not None:
        print(f"[Main] Task model params: {task_cfg.get('params')}")

    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_0", size=640 * 640 * 3)
    shm_manager.create_block("shm_task", size=640 * 640 * 3)
    shm_manager.create_block("shm_lane", size=8)
    shm_manager.create_block("shm_crop", size=640 * 640 * 3)
    request_queue = multiprocessing.Queue()
    publish_queue = multiprocessing.Queue()
    action_done_event = multiprocessing.Event()

    mgr = ProcessManager(setup_signal_handlers=False)

    print("[Main] Starting UART service process...")
    mgr.add_process(
        target=serial_server_process,
        args=(SERIAL_PHYSICAL_PATH, 115200, request_queue, publish_queue, action_done_event),
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
        action_done_event=action_done_event,
    )
    brain.start()

    try:
        while not shutdown_event.wait(timeout=1.0):
            pass
    except KeyboardInterrupt:
        request_shutdown(signal.SIGINT)
    finally:
        print("[Main] Cleaning up...")
        if action_done_event is not None:
            action_done_event.set()
        if brain is not None:
            brain.cleanup()
        if mgr is not None:
            mgr.terminate_all()
        if shm_manager is not None:
            shm_manager.release_all()
        print("[Main] Shutdown complete.")
        sys.exit(0)


if __name__ == "__main__":
    start_framework()
