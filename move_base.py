import time
import Zmq_mod
from shared_memory_manager import SharedMemoryManager

class Base_func:
    """底盘与云台的控制基石，负责去 ZMQ Pub 提取硬件状态和发送命令"""
    def __init__(self, mode='normal'):
        self.mode = mode
        
        # ⚠️ 修改点 1：不要在主线程初始化 ZMQ Socket！
        # 我们把它留到 MOD_LANE 实际被调用的子线程里去懒加载
        self.zmq_req = None 
        
        # 共享内存管理器，用于从中读取巡线网络输出的结果
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_block("shm_lane", size=8) # 两个 float 8 字节

    def MOD_LANE(self, base_speed=-0.28):
        """核心闭环巡航动作函数！"""
        
        # ⚠️ 修改点 2：懒加载 ZMQ Socket，确保它和调用者在同一个线程内！
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode='req', address='ipc:///tmp/MainWithUart.ipc')

        try:
            # 瞬间(O(1))从内存读取相机进程解算出的 2 个 float
            angle, infer_speed = self.shm_manager.read_floats("shm_lane", 2)
            
            if angle > 120.0: angle = 120.0
            if angle < -120.0: angle = -120.0

            target_speed = base_speed
            
            print(f"[MoveBase] MOD_LANE -> Executing Deviation: {angle:.2f}, Speed: {target_speed:.2f}")

            data = {
                "cmd": "Motion",
                "mode": 0,                
                "deviation": float(angle),
                "speed_x": float(target_speed) 
            }
            
            self.zmq_req.send(data)

            # 等待串口进程接受请求的回执
            reply = self.zmq_req.receive()
            
            # ⚠️ 修改点 3：打破 REQ 死锁！如果超时，必须断臂求生，重建 Socket
            if reply is None:
                print("[MoveBase] 🚨 ZMQ REQ 接收超时！正在重建 Socket 防止状态机死锁...")
                self.zmq_req.close()
                self.zmq_req = None  # 下一帧会自动重新连接
            elif reply.get("status") != "ok":
                print(f"[MoveBase] Warning: UART Server returned error: {reply}")
                
        except Exception as e:
            print(f"[MoveBase] Lane Following Exec Error: {e}")
            # 发生未知异常时，同样重置 ZMQ
            if self.zmq_req:
                self.zmq_req.close()
                self.zmq_req = None

    def MOD_STOP(self):
        """强制急停"""
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode='req', address='ipc:///tmp/MainWithUart.ipc')
        data = {"cmd": "Motion", "mode": 1, "pos_x": 0, "pos_y": 0, "z_angle": 0}
        self.zmq_req.send(data)
        self.zmq_req.receive()

    def execute_arm_motion(self, mot0, mot1, mot2, mot3, suck=0, light=0):
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode='req', address='ipc:///tmp/MainWithUart.ipc')
        data = {"cmd": "Arm", "mot0": mot0, "mot1": mot1, "mot2": mot2, "mot3": mot3, "suck": suck, "light": light}
        self.zmq_req.send(data)
        return self.zmq_req.receive()

    def set_sys_mode(self, flag):
        if self.zmq_req is None:
            self.zmq_req = Zmq_mod.ZMQComm(mode='req', address='ipc:///tmp/MainWithUart.ipc')
        data = {"cmd": "SysMode", "flag": flag}
        self.zmq_req.send(data)
        return self.zmq_req.receive()

class Task_func:
    """ 
    多任务执行库 (占位符)
    对应报告中预留的四个任务槽位
    """
    def __init__(self, base_func: Base_func):
        self.base = base_func

    def task1_executor(self):
        """ Task 1: 汉诺塔 (Hanoi) 接口预留 """
        print("[Task_func] Executing Task 1 (Hanoi) Placeholder...")
        pass

    def task2_executor(self):
        """ Task 2: 大模型交互 (LLM/BMI) 接口预留 """
        print("[Task_func] Executing Task 2 (LLM) Placeholder...")
        pass

    def task3_executor(self):
        """ Task 3: 视觉伺服 (Visual Servoing) 接口预留 """
        print("[Task_func] Executing Task 3 (Servoing) Placeholder...")
        pass

    def task4_executor(self):
        """ Task 4: 保留任务接口 """
        print("[Task_func] Executing Task 4 (Reserved) Placeholder...")
        pass