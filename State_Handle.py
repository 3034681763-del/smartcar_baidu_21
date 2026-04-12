import threading
import time
from move_base import Base_func
from shared_memory_manager import SharedMemoryManager
import Zmq_mod
import numpy as np
from infer_server_client import InferClient1
class TaskManager:
    """ 
    大本营进程中的守护状态机 
    双核驱动: 判断线程（judge) + 执行线程 (handle)
    """
    def __init__(self):
        # 初始化控制基础实例，附带了通信的挂载
        self.movebase = Base_func()
        
        # 预留的四个任务处理库
        from move_base import Task_func
        self.task_func = Task_func(self.movebase)
        
        # 定义状态机: 预留 Task1, Task2, Task3, Task4 等任务插槽
        self.current_task = "Lane" 
        
        # 各个并发守护旗标
        self.stopJudge = threading.Event()
        self.stopHandle = threading.Event()

        # 底盘环境反馈订阅客户端 (SUB)
        self.zmq_sub_info = Zmq_mod.ZMQComm(mode='sub', address='ipc:///tmp/SubUartInfo.ipc')
        
        # ======= 【内存大管理】 ======= #
        # 创建这 3 个全局管控的核心零拷贝内存，因为他们在主程存活最长
        self.shm_manager = SharedMemoryManager()
        # 1. 摄像头捕获原图 (640x480x3) -> 可给别的进程显示用
        self.shm_manager.create_block("shm_0", size=640 * 640 * 3) 
        # 2. 从 AI 那拿出的两枚 Float (偏差、油门)
        self.shm_manager.create_block("shm_lane", size=8)

        # ====== 【多任务通信层】 ====== #
        # 预留视觉任务 RPC 客户端 (端口需与服务器对齐，例如 5020)
        self.task_client = InferClient1('task', self.shm_manager, 5020)

        # 缓存一下串口下发的各类传感器值
        self.current_sensor_data = {}

    def judge_task_thread(self):
        """ 线程 1: 环境感知与状态跃迁核心 """
        print("[TaskManager] Judge Thread Started.")
        no_data_count = 0
        
        while not self.stopJudge.is_set():
            time.sleep(0.02) # 50Hz 获取底盘环境
            # 这里以非阻塞方式快速消耗订阅到的底盘数据
            sub_msg = self.zmq_sub_info.receive()
            if sub_msg and isinstance(sub_msg, dict):
                no_data_count = 0
                if sub_msg.get("cmd") == "PushResp":
                    self.current_sensor_data = sub_msg.get("data", {})
                    # 例如，如果需要判断距离传感器 < 10cm 则强制改变状态为 "Stop" 可以写在这里
                    # tofR = self.current_sensor_data.get("dist_sensorR", 999)
                    # if tofR < 100: self.current_task = "Stop"
            else:
                no_data_count += 1
                if no_data_count > 100: # 长时间未收到串口包，可能有异常
                    pass 
            
            # --- 【预留的视觉辅助决策层】 ---
            # 如果当前在巡线，检查下有没有触发特定任务
            # if self.current_task == "Lane":
            #     results = self.task_client("shm_0", (480, 640, 3), np.uint8)
            #     if results and any(r.label == "tower" for r in results):
            #         self.current_task = "Task1"

    def handle_task_thread(self):
        """ 线程 2: 死板的任务流水线执行机 """
        print("[TaskManager] Handler Thread Started.")
        
        while not self.stopHandle.is_set():
            time.sleep(0.05) # 执行帧率约 20 FPS，与深度模型的速度保持一致即可
            
            task = self.current_task
            if task == "Lane":
                # 去提取内存模型预测发给串口
                self.movebase.MOD_LANE(base_speed=-0.28)
            elif task == "Task1":
                self.task_func.task1_executor()
                self.current_task = "Lane"  # 模拟执行完后回到巡线
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
        """挂载所有守护线程"""
        self.t_judge = threading.Thread(target=self.judge_task_thread, daemon=True)
        self.t_handle = threading.Thread(target=self.handle_task_thread, daemon=True)
        self.t_judge.start()
        self.t_handle.start()

    def cleanup(self):
        """资源终结器"""
        print("[TaskManager] Cleaning up Sub Threads and Shared Memory...")
        self.stopJudge.set()
        self.stopHandle.set()
        self.movebase.MOD_STOP()
        self.zmq_sub_info.close()
        self.shm_manager.release_all()
