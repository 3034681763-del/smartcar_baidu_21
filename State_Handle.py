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

    def __init__(
        self,
        request_queue=None,
        publish_queue=None,
        enable_aux_models=True,
        task_shm_key="shm_task",
    ):
        self.movebase = Base_func(request_queue=request_queue)

        self.current_task = "Lane"
        self.stopJudge = threading.Event()
        self.stopHandle = threading.Event()
        self.publish_queue = publish_queue
        self.task_shm_key = task_shm_key
        self.fache = 0
        self.cheat_flag = 0
        self.tofL = 5000
        self.tofR = 5000
        self.world_y = 0
        self.angz = 0
        self.seed_entry_count = 0
        self.seed_entry_frames = 3

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_block("shm_0", size=640 * 640 * 3)
        self.shm_manager.create_block(self.task_shm_key, size=640 * 640 * 3)
        self.shm_manager.create_block("shm_lane", size=8)
        self.shm_manager.create_block("shm_crop", size=640 * 640 * 3)

        self.task_clients = {}
        self.task_client = None
        self.ocr_reader = None
        self.enable_aux_models = enable_aux_models

        if self.enable_aux_models:
            for cfg in model_configs:
                self.task_clients[cfg["name"]] = InferClient1(cfg["name"], self.shm_manager, cfg["port"])

            self.task_client = self.task_clients["task"]
            self.ocr_reader = OCRPipeline(
                self.shm_manager,
                self.task_clients["task"],
                self.task_clients["word"],
                crop_key="shm_crop",
            )
            print("[TaskManager] Auxiliary model clients are ready.")
        else:
            print("[TaskManager] Auxiliary model clients are disabled.")

        from move_base import Task_func

        self.task_func = Task_func(
            self.movebase,
            self.ocr_reader,
            task_shm_key=self.task_shm_key,
            task_client=self.task_client,
        )
        self.current_sensor_data = {}

    def read_text_once(self, result_amount=1, sort_by="y"):
        if self.ocr_reader is None:
            return []
        return self.ocr_reader.read_texts(
            shm_key=self.task_shm_key,
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
                    self.fache = self.current_sensor_data.get("Fache_flag", 0)
                    self.cheat_flag = self.current_sensor_data.get("Cheat_flag", 0)
                    self.tofL = self.current_sensor_data.get("dist_sensorL", 5000)
                    self.tofR = self.current_sensor_data.get("dist_sensorR", 5000)
                    self.world_y = self.current_sensor_data.get("world_y", 0)
                    self.angz = self.current_sensor_data.get("world_z_angle", 0)

                    if self.current_task == "Lane":
                        if self.world_y > 10000 and self.tofR < 100:
                            self.seed_entry_count += 1
                            if self.seed_entry_count >= self.seed_entry_frames:
                                print("[TaskManager] Enter Task1 seeding.")
                                self.current_task = "Task1"
                                self.seed_entry_count = 0
                        else:
                            self.seed_entry_count = 0
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
            task_key = task.lower() if isinstance(task, str) else ""
            if task_key == "lane":
                self.movebase.MOD_LANE(base_speed=-0.28)
            elif task_key == "tracking":
                self.task_func.tracking_executor()
            elif task_key == "basetest":
                self.task_func.base_motion_test_executor()
                self.current_task = "Lane"
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
