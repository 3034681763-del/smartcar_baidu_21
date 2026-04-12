import Zmq_mod
from shared_memory_manager import SharedMemoryManager


class Base_func:
    """Base control bridge for chassis and actuator commands."""

    def __init__(self, mode="normal"):
        self.mode = mode
        self.zmq_req = None
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_block("shm_lane", size=8)

    def MOD_LANE(self, base_speed=-0.28):
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode="req", address="ipc:///tmp/MainWithUart.ipc")

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
            self.zmq_req.send(data)
            reply = self.zmq_req.receive()

            if reply is None:
                print("[MoveBase] ZMQ timeout, rebuilding REQ socket...")
                self.zmq_req.close()
                self.zmq_req = None
            elif reply.get("status") != "ok":
                print(f"[MoveBase] Warning: UART Server returned error: {reply}")
        except Exception as exc:
            print(f"[MoveBase] Lane Following Exec Error: {exc}")
            if self.zmq_req:
                self.zmq_req.close()
                self.zmq_req = None

    def MOD_STOP(self):
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode="req", address="ipc:///tmp/MainWithUart.ipc")
        data = {"cmd": "Motion", "mode": 1, "pos_x": 0, "pos_y": 0, "z_angle": 0}
        self.zmq_req.send(data)
        self.zmq_req.receive()

    def execute_arm_motion(self, mot0, mot1, mot2, mot3, suck=0, light=0):
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode="req", address="ipc:///tmp/MainWithUart.ipc")
        data = {
            "cmd": "Arm",
            "mot0": mot0,
            "mot1": mot1,
            "mot2": mot2,
            "mot3": mot3,
            "suck": suck,
            "light": light,
        }
        self.zmq_req.send(data)
        return self.zmq_req.receive()

    def set_sys_mode(self, flag):
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode="req", address="ipc:///tmp/MainWithUart.ipc")
        data = {"cmd": "SysMode", "flag": flag}
        self.zmq_req.send(data)
        return self.zmq_req.receive()


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
