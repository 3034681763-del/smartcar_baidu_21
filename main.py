import sys
import time

from Cam_cap import vidpub_course
from Process_manage import ProcessManager
from SerialCommunicate import serial_server_process
from State_Handle import TaskManager
from infer_server_client import model_configs, serve_model_process
from shared_memory_manager import SharedMemoryManager


def start_framework():
    print("=" * 60)
    print(" SmartCar 21 Lane Framework With OCR Support ")
    print("=" * 60)

    shm_manager = SharedMemoryManager()
    shm_manager.create_block("shm_0", size=640 * 640 * 3)
    shm_manager.create_block("shm_lane", size=8)
    shm_manager.create_block("shm_crop", size=640 * 640 * 3)

    mgr = ProcessManager()

    print("[Main] Starting UART service process...")
    mgr.add_process(target=serial_server_process, args=("1-2.1:1.0",))

    print("[Main] Starting lane camera process...")
    mgr.add_process(
        target=vidpub_course,
        args=("shm_0", "shm_lane", "model.nb", "1-2.2:1.0"),
    )

    print("[Main] Starting task and OCR model services...")
    for cfg in model_configs:
        mgr.add_process(target=serve_model_process, args=(cfg, shm_manager))

    mgr.start_all()

    time.sleep(1.5)

    print("[Main] Starting state manager...")
    brain = TaskManager()
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
