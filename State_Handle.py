from queue import Empty
import threading
import time

import numpy as np

from infer_server_client import InferClient1, OCRPipeline, model_configs
from move_base import Base_func
from shared_memory_manager import SharedMemoryManager


class TaskManager:
    """
    Main state machine with a judge thread and a handler thread.
    """

    def __init__(self, request_queue=None, publish_queue=None):
        self.movebase = Base_func(request_queue=request_queue)

        self.current_task = "Lane"
        self.stopJudge = threading.Event()
        self.stopHandle = threading.Event()
        self.publish_queue = publish_queue

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_block("shm_0", size=640 * 640 * 3)
        self.shm_manager.create_block("shm_lane", size=8)
        self.shm_manager.create_block("shm_crop", size=640 * 640 * 3)

        self.task_clients = {}
        for cfg in model_configs:
            self.task_clients[cfg["name"]] = InferClient1(cfg["name"], self.shm_manager, cfg["port"])

        self.task_client = self.task_clients["task"]
        self.ocr_reader = OCRPipeline(
            self.shm_manager,
            self.task_clients["task"],
            self.task_clients["word"],
            crop_key="shm_crop",
        )

        from move_base import Task_func

        self.task_func = Task_func(self.movebase, self.ocr_reader)
        self.current_sensor_data = {}

    def read_text_once(self, result_amount=1, sort_by="y"):
        return self.ocr_reader.read_texts(
            shm_key="shm_0",
            shape=(640, 640, 3),
            dtype=np.uint8,
            target_label="text",
            result_amount=result_amount,
            sort_by=sort_by,
        )

    def judge_task_thread(self):
        print("[TaskManager] Judge Thread Started.")
        no_data_count = 0

        while not self.stopJudge.is_set():
            try:
                sub_msg = self.publish_queue.get(timeout=0.02) if self.publish_queue else None
            except Empty:
                sub_msg = None

            if sub_msg and isinstance(sub_msg, dict):
                no_data_count = 0
                if sub_msg.get("cmd") == "PushResp":
                    self.current_sensor_data = sub_msg.get("data", {})
            else:
                no_data_count += 1
                if no_data_count > 100:
                    pass

            # Example OCR trigger point for future task logic:
            # if self.current_task == "Task2":
            #     texts = self.read_text_once(result_amount=1)
            #     if texts:
            #         print(f"[TaskManager] OCR text: {texts}")

    def handle_task_thread(self):
        print("[TaskManager] Handler Thread Started.")

        while not self.stopHandle.is_set():
            time.sleep(0.05)
            task = self.current_task
            if task == "Lane":
                self.movebase.MOD_LANE(base_speed=-0.28)
            elif task == "Task1":
                self.task_func.task1_executor()
                self.current_task = "Lane"
            elif task == "Task2":
                self.task_func.task2_executor()
                self.current_task = "Lane"
            elif task == "Task3":
                self.task_func.task3_executor()
                self.current_task = "Lane"
            elif task == "Task4":
                self.task_func.task4_executor()
                self.current_task = "Lane"
            elif task == "Stop":
                self.movebase.MOD_STOP()
                time.sleep(1)

    def start(self):
        self.t_judge = threading.Thread(target=self.judge_task_thread, daemon=True)
        self.t_handle = threading.Thread(target=self.handle_task_thread, daemon=True)
        self.t_judge.start()
        self.t_handle.start()

    def cleanup(self):
        print("[TaskManager] Cleaning up threads and shared memory...")
        self.stopJudge.set()
        self.stopHandle.set()
        self.movebase.MOD_STOP()
        for client in self.task_clients.values():
            client.close()
        self.shm_manager.release_all()
