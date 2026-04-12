import serial
import struct
import time
from threading import Thread
import Zmq_mod 

# =============================================================================
# 一、串口通讯底层驱动
# =============================================================================
class SerialCommunicate:
    def __init__(self, physical_path, baudrate):
        serial_device = self.find_serial_by_path(physical_path)
        if serial_device:
            print(f"[UART] 连接设备成功：{serial_device}")
            try:
                self.ser = serial.Serial(serial_device, baudrate, timeout=0.5)
                if not self.ser.isOpen():
                    self.ser.open()
            except Exception as e:
                print("[UART] 串口打开异常：", e)
                self.ser = None
                return
        else:
            print(f"[UART] 未找到设备：{physical_path}")
            self.ser = None
            return

        self.last_response = {}
        # 开启后台守护线程不断收包
        self.thread_recv = Thread(target=self.get_data_thread, args=(), daemon=True)
        self.thread_recv.start()

    def find_serial_by_path(self, physical_path):
        import pyudev
        context = pyudev.Context()
        for device in context.list_devices(subsystem='usb-serial'):
            path = device.device_path.rsplit('/', 1)[0]
            if path.endswith(physical_path):
                return '/dev/' + device.device_path.rsplit('/', 1)[1]
        return None

    def get_data_thread(self):
        """ 解析底层上传的数据字典，剥离原版多余代码，保留结构体定长还原流 """
        while True:
            if self.ser.in_waiting > 0:
                try:
                    start = self.ser.read(1)
                    if struct.unpack('=b', start)[0] != 0x42:
                        continue
                    
                    tmpData = [start]
                    address = self.ser.read(1)
                    tmpData.append(address)

                    len_byte = self.ser.read(1)
                    if not len_byte: continue
                    tmpData.append(len_byte)

                    frameLen = struct.unpack('=b', len_byte)[0] 
                    remaining = frameLen - 3  
                    for i in range(remaining):
                        tmpData.append(self.ser.read(1))
                    
                    if struct.unpack('=b', tmpData[frameLen -1])[0] != 0x3c:
                        continue 

                    data_field = tmpData[3:frameLen-1] 
                    cmd = struct.unpack('=b', tmpData[1])[0] 

                    if cmd == 0x01:
                        if len(data_field) == 18:
                            byte_data = b''.join(data_field)
                            facheflag, cheatflag, dist_sensorl, dist_sensorr, world_y, world_z_angle = \
                                struct.unpack('=BBffff', bytes(byte_data))
                                
                            self.last_response = {
                                "Fache_flag": facheflag,
                                "Cheat_flag": cheatflag,
                                "dist_sensorL": dist_sensorl,
                                "dist_sensorR": dist_sensorr,
                                "world_y": world_y,
                                "world_z_angle": world_z_angle
                            }
                except Exception as e:
                    print(f"数据接收异常：{e}")
                    continue

    def send_motion_mode(self, mode, deviation=None, speed_x=None, pos_x=None, pos_y=None, z_angle=None):
        """ 发送运动模式命令（命令ID=0x11） """
        if self.ser is None: return

        if mode == 0:
            if deviation is None or speed_x is None: return
            # 兼容冠军板格式: =BBBB2fB -> 6 Byte + 2 float -> len 13
            packet = struct.pack('=BBBB2fB', 0x42, 0x11, 13, 0, float(speed_x), float(deviation), 0x3c)
            self.ser.write(packet)
        elif mode == 1:
            if pos_x is None or pos_y is None or z_angle is None: return
            packet = struct.pack('=BBBB3fB', 0x42, 0x11, 17, 1, float(pos_x), float(pos_y), float(z_angle), 0x3c)
            self.ser.write(packet)

    def send_arm_motion(self, mot0, mot1, mot2, mot3, suck, light):
        """ 发送机械臂运动命令（命令ID=0x12） """
        if self.ser is None: return
        # 遵循冠军版协议格式: =3b4h3b
        packet = struct.pack('=3b4h3b', 0x42, 0x12, 14, int(mot0), int(mot1), int(mot2), int(mot3), int(suck), int(light), 0x3c)
        self.ser.write(packet)

    def send_mode_flag(self, flag):
        """ 发送系统模式标志命令（命令ID=0x13） """
        if self.ser is None: return
        # 遵循冠军版协议格式: =BBBBB
        packet = struct.pack('=BBBBB', 0x42, 0x13, 0x05, int(flag), 0x3c)
        self.ser.write(packet)

    def close(self):
        if self.ser and self.ser.isOpen():
            self.ser.close()

# =============================================================================
# 二、服务端网络包装层
# =============================================================================
class SerialServer:
    """提供 ZeroMQ REQ 和 PUB 微服务的代理器"""
    def __init__(self, serial_path="1-2.1:1.0", baudrate=115200):
        self.serial_comm = SerialCommunicate(serial_path, baudrate)
        
        # 将 ZMQ 端点配置为文件系统路径 (ipc) 实现跨进程高速响应
        self.zmq_req = Zmq_mod.ZMQComm(mode='rep', address='ipc:///tmp/MainWithUart.ipc')
        self.zmq_pub = Zmq_mod.ZMQComm(mode='pub', address='ipc:///tmp/SubUartInfo.ipc')
        self.flag_exit = False

    def process_rep_requests(self):
        """ 接收来自上位机 StateMachine 下发的执行指令 """
        while not self.flag_exit:
            try:
                req = self.zmq_req.receive()
                if not req: continue
                
                cmd_type = req.get("cmd")
                if cmd_type == "Motion":
                    mode = req.get("mode")
                    if mode == 0:
                        self.serial_comm.send_motion_mode(0, deviation=req["deviation"], speed_x=req["speed_x"])
                    elif mode == 1:
                        self.serial_comm.send_motion_mode(1, pos_x=req["pos_x"], pos_y=req["pos_y"], z_angle=req["z_angle"])
                
                elif cmd_type == "Arm":
                    self.serial_comm.send_arm_motion(
                        req["mot0"], req["mot1"], req["mot2"], req["mot3"], 
                        req.get("suck", 0), req.get("light", 0)
                    )
                
                elif cmd_type == "SysMode":
                    self.serial_comm.send_mode_flag(req["flag"])

                # 告知上层接收成功
                self.zmq_req.send({"status": "ok"})
            except Exception as e:
                print(f"[SerialServer] Request Error: {e}")
                pass

    def pub_loop(self):
        """ 不断往外高频广播底盘坐标与传感信息 """
        while not self.flag_exit:
            data = self.serial_comm.last_response 
            if data:
                self.zmq_pub.send({"cmd": "PushResp", "data": data})
            time.sleep(0.02) # 50Hz 刷新率

    def run(self, stop_event=None):
        t1 = Thread(target=self.process_rep_requests, daemon=True)
        t2 = Thread(target=self.pub_loop, daemon=True)
        t1.start()
        t2.start()
        
        # 阻塞并响应主进程结束信号
        if stop_event:
            stop_event.wait()
            print("[SerialServer] 接到终止信号，开始关闭串口...")
            self.close()

    def close(self):
        self.flag_exit = True
        self.zmq_req.close()
        self.zmq_pub.close()
        self.serial_comm.close()

# 提供给进程调用方的入口函数
def serial_server_process(stop_event, physical_path="1-2.1:1.0"):
    print("[SerialServer] 硬件通讯子进程启动...")
    server = SerialServer(serial_path=physical_path)
    server.run(stop_event)
