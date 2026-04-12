from shared_memory_manager import SharedMemoryManager


class Base_func:
    """Base control bridge for chassis and actuator commands."""

    def __init__(self, mode="normal", request_queue=None):
        self.mode = mode
        self.request_queue = request_queue
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_block("shm_lane", size=8)

    def MOD_LANE(self, base_speed=-0.28):
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

    def set_sys_mode(self, flag):
        data = {"cmd": "SysMode", "flag": flag}
        if self.request_queue is not None:
            self.request_queue.put(data)
        return {"status": "queued", "cmd": "SysMode"}


class Task_func:
    """Placeholder task library. Task2 now demonstrates the OCR call chain."""

    def __init__(self, base_func: Base_func, ocr_reader=None):
        self.base = base_func
        self.ocr_reader = ocr_reader

    def task1_executor(self):
        print("[Task_func] Executing Task 1 placeholder...")

    def task2_executor(self):
        print("[Task_func] Executing Task 2 OCR placeholder...")
        if self.ocr_reader is None:
            print("[Task_func] OCR reader is not configured.")
            return

        texts = self.ocr_reader.read_texts(
            shm_key="shm_0",
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
