import psutil
import os
import subprocess
import sys

import multiprocessing
import signal
import atexit
import time
from typing import Callable, List, Tuple

# brief：子进程管理，在主进程退出时，通知子进程关闭，如果3s没有关闭，强制杀死
# 注意使用时，子进程函数的第一个参数必须是multiprocessing.Event

class ProcessManager:
    def __init__(self):
        self.processes = []
        self._setup_exit_handlers()

    def add_process(self, target: Callable, args: tuple = ()):
        """
        添加子进程任务，target 函数的第一个参数必须是 stop_event
        """
        stop_event = multiprocessing.Event()
        full_args = (stop_event,) + args
        p = multiprocessing.Process(target=target, args=full_args)
        # 不使用 daemon 模式，避免子进程功能受限
        self.processes.append((p, stop_event))

    def start_all(self):
        for p, _ in self.processes:
            p.start()

    def terminate_all(self):
        for p, stop_event in self.processes:
            if p.is_alive():
                stop_event.set()  # 通知子进程退出
        for p, _ in self.processes:
            p.join(timeout=3)  # 给子进程最多3秒收尾
            if p.is_alive():
                p.terminate()  # 仍未退出则强制杀掉
                p.join()

    def _setup_exit_handlers(self):
        atexit.register(self.terminate_all)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[ProcessManager] Caught signal {signum}, terminating subprocesses...")
        self.terminate_all()
        exit(0)

def get_python_processes():
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if len(cmdline) > 1:
                    script_path = cmdline[1]
                    python_processes.append((proc.info['pid'], script_path))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return python_processes

"""前台开启进程,输出会打印在终端中"""
def check_back_python(file_name):
    dir_file = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(dir_file, file_name)
    if not os.path.exists(file_path):
        raise Exception(f"文件不存在: {file_path}")
    
    py_processes = get_python_processes()
    found = False
    
    for pid, script_path in py_processes:
        script_name = os.path.basename(script_path)
        if script_name == file_name:
            found = True
            break
    
    if not found:
        cmd = [sys.executable, file_path]
        # 直接显示输出到当前终端（不静默）
        subprocess.Popen(cmd)  # 移除 stdout/stderr 重定向
        print(f"已启动: {file_name}（输出显示在本终端）")
    else:
        print(f"已在运行: {file_name}")
