import time
import sys
from Process_manage import ProcessManager
from SerialCommunicate import serial_server_process
from Cam_cap import vidpub_course
from State_Handle import TaskManager

def start_framework():
    print("=" * 60)
    print(" 智能汽车 21 届备战 - 边缘计算循迹专属框架 (冠军版1:1完整还原) ")
    print("=" * 60)
    
    # 初始化多进程管家
    mgr = ProcessManager()

    # 1. 注册硬件串口通讯 RPC 集群服务
    print("[Main] 注册串口与底盘网关服务子进程...")
    # 请填写真实的 Uart 口（如 1-2.1:1.0 / ttyUSB0）
    mgr.add_process(target=serial_server_process, args=("1-2.1:1.0",))
    
    # 2. 注册端侧零拷贝视觉流微服务
    print("[Main] 注册最高优先级：端到端深度视觉反馈子进程...")
    # （这里的模型路径如果是真的，需要指定正确的 nb文件）
    mgr.add_process(target=vidpub_course, args=("shm_0", "shm_lane", "model.nb", "1-2.2:1.0"))
    
    # => 点火！拔除操作系统对以上进程的干涉，接管生命周期
    mgr.start_all()
    
    time.sleep(1) # 等待共享内存与 IPC 构建 Socket 连接文件
    
    # 3. 注册主控状态机
    print("[Main] 挂载高层状态机，执行决策！")
    brain = TaskManager()
    brain.start()
    
    try:
        # 维持主进程存活运行，所有工作由底层 ZMQ、Thread 和各进程默默高速传递
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C 收到！主程正在命令各集群停止收割...")
        brain.cleanup()
        mgr.terminate_all()
        print("[Main] 安全且优雅地停止完毕。")
        sys.exit(0)

if __name__ == '__main__':
    start_framework()
