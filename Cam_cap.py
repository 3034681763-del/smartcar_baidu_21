import numpy as np
from multiprocessing import shared_memory
import cv2
import time
import pyudev
import copy
from paddlelite.lite import *
import struct

class CameraCapture:
    def __init__(self, camera_path_0):
        self.camera_path_0 = camera_path_0
        
        # 共享内存指针
        self.shm_image_0 = None
        self.shm_lane_ret_bytes = None
        
        # 硬件与模型指针
        self.cap_0 = None
        self.paddle_predictor = None

    def find_camera_by_path(self, physical_path):
        """利用 pyudev 根据物理 USB hub 路径锁定固定 /dev/videoX，防止每次开机序号飘移"""
        context = pyudev.Context()
        for device in context.list_devices(subsystem='video4linux'):
            # 找到 /video4linux 之前的 USB 路径
            end_index = device.device_path.rfind("/video4linux")
            if end_index != -1:
                cleaned_string = device.device_path[:end_index]
            else:
                cleaned_string = device.device_path

            if cleaned_string.endswith(physical_path):
                return '/dev/' + device.device_path.rsplit('/', 1)[1]
        return None

    def setup_shared_memory(self, shm_img_name, shm_lane_name):
        """挂载已经在 Main 进程开辟好的共享内存"""
        self.shm_image_0 = shared_memory.SharedMemory(name=shm_img_name)
        self.shm_lane_ret_bytes = shared_memory.SharedMemory(name=shm_lane_name)

    def setup_paddle_predictor(self, model_path):
        """配置飞桨 Edge 端侧轻量化推理引擎"""
        try:
            config = MobileConfig() # type: ignore
            config.set_model_from_file(model_path)
            self.paddle_predictor = create_paddle_predictor(config) #type:ignore
            print(f"[CamCapture] 模型 {model_path} 加载成功！")
        except Exception as e:
            print(f"[CamCapture] PaddleLite 引擎初始化失败，可能未安装或模型路径错误。请勿在无模型下用于真实比赛。错误名: {e}")

    def open_camera(self):
        """开启物理摄像头并设置最高效的采集分辨率"""
        dev_0 = self.find_camera_by_path(self.camera_path_0)
        if dev_0 is None:
            print(f"[CamCapture] Camera 0 with path {self.camera_path_0} not found.")
            # return False # 放开这行用于无设备单机测试
            dev_0 = 0 # 没有特定的 USB 时， fallback 到默认 0 号设备
            
        self.cap_0 = cv2.VideoCapture(dev_0)
        if not self.cap_0.isOpened():
             print("[CamCapture] 无法打开物理摄像头！")
             return False

        self.cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        self.cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
        
        print("[CamCapture] 摄像头硬件就绪。")
        return True
    
    def process_frame(self, frame, shm_image_array):
        """深拷贝图像，防止异步读写导致内存撕裂"""
        frame_copy = copy.deepcopy(frame)
        shm_image_array[:] = np.array(frame_copy)[:]

    def run_inference(self, frame):
        """
        真正的极简端到端推理闭环：
        图片 -> 归一化 -> 送入模型 -> 吐出两个 float 回归值 (Deviation偏差, Speed推荐速度)
        """
        if self.paddle_predictor is None:
            return np.array([[0.0, 0.0]]) # 测试环境空转

        # 1. 深度学习极度压缩前处理 (HWC 128x128)
        img = cv2.resize(frame, (128, 128))
        # 归一化到 [-1, 1] 之间
        img = (img.astype(np.float32) - 127.5) / 127.5
        # BGR格式 翻转为 RGB
        img = img[:, :, ::-1].astype('float32')  
        # HWC (高宽通道) 转化为 CHW (通道宽高) Tensor 标准
        img = img.transpose((2, 0, 1))  
        # 扩充 Batch 维度： 1xCxHxW
        image_data = img[np.newaxis, :]
        
        # 2. 从边缘计算设备 NPU/CPU 推理并拿回结果
        input_tensor = self.paddle_predictor.get_input(0)
        input_tensor.from_numpy(image_data)
        self.paddle_predictor.run()
        
        # 预期输出为 [[float1, float2]]
        return self.paddle_predictor.get_output(0).numpy() 

    def run(self, stop_event):
        """
        高能核心循环：以跑满单核的速度，疯狂吃图、算出偏差、以 O(1) 的时间写入内存堆！
        """       
        start_time = time.time()
        frame_count = 0
               
        print("[CamCapture] 循迹摄像与高速端侧推理死循环已启动...")
        while not stop_event.is_set():
           
            ret_0, frame_0 = self.cap_0.read()
            
            # FPS 统计
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0: 
                fps0 = frame_count / elapsed_time
                print(f"[CamCapture] 实时感知推理帧率: {fps0:.2f} FPS")
                frame_count = 0
                start_time = time.time()
                
            if ret_0:
                frame_count += 1
                # 为了保持全局原图分辨率清晰度，resize 到 640 （模型会在 run_inference 内部再缩小）
                frame_0 = cv2.resize(frame_0, (640, 640), interpolation=cv2.INTER_AREA)

                # ================= 核心闭环：模型计算 =================
                # lane_infer_ret [0][0] 为角度偏差, [0][1] 为油门深浅
                lane_infer_ret = self.run_inference(frame_0)
                
                # 乘系数 1.1 作为赛道调参增益
                lane_infer_ret[0][0] *= 1.1 

                # ================= 核心闭环：低延迟打包写入 =================
                # 把两个 float 浮点数打包成紧实的 8 Bytes！
                ret_bytes = struct.pack('=ff', lane_infer_ret[0][0], lane_infer_ret[0][1])
                # 直接通过共享内存内存指针写穿（零拷贝），供 StateMachine 与底层获取!
                self.shm_lane_ret_bytes.buf[0:len(ret_bytes)] = ret_bytes

                # ================= 附加服务 =================
                # 把原始高清图抛进内存，以便给将来想要加上检测冰箱/抓取服务的别的 YOLO 进程去吃
                self.process_frame(frame_0, np.ndarray(frame_0.shape, dtype=frame_0.dtype, buffer=self.shm_image_0.buf))
                
        # 收尾退出
        print("[CamCapture] 接到终止信号，正在注销...")
        if self.cap_0 is not None:
             self.cap_0.release()

def vidpub_course(stop_event, shm0, shm_lane, model_path="src/cnn_auto.nb", camera_path="1-2.2:1.0"):   
    """
    提供给 ProcessManager 的子进程入口函数
    """
    # 实例化感知引擎
    camera_capture = CameraCapture(camera_path_0=camera_path) 
    
    # 挂载由主进程已经申请好的内存块
    camera_capture.setup_shared_memory(shm_img_name=shm0, shm_lane_name=shm_lane)
    
    # 初始化边缘算力引擎 (.nb 格式)
    camera_capture.setup_paddle_predictor(model_path=model_path)
    
    # 如果打得开物理相机，就锁死在这个强烈的 While 循环里推流！
    if camera_capture.open_camera():
        camera_capture.run(stop_event)
