# !/usr/bin/env python3
import multiprocessing
import socket
import struct
import pickle
import threading
import time
import select
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from shared_memory_manager import SharedMemoryManager
import os

# ----------------- 辅助函数 -----------------
def send_pickle(conn: socket.socket, obj):
    """使用 socket.makefile 发送序列化对象"""
    try:
        file = conn.makefile('wb')
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
    except Exception as e:
        raise RuntimeError(f"send_pickle 失败: {e}")

def recv_pickle(conn: socket.socket):
    """使用 socket.makefile 接收序列化对象"""
    try:
        file = conn.makefile('rb')
        obj = pickle.load(file)
        return obj
    except Exception as e:
        raise RuntimeError(f"recv_pickle 失败: {e}")

# ----------------- 客户端 -----------------
class InferClient1:
    def __init__(self, model_name: str, shm, port: int, timeout=2):
        self.model_name = model_name
        self.shm = shm
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()
        self.reconnect_interval = 1
        # 阻塞等待服务器端就绪（可选，通常在任务开始前调用一次即可）
        # self._block_until_server_ready()

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(self.timeout)
            self.sock.connect(('localhost', self.port))
            return True
        except Exception as e:
            self.sock = None
            return False

    def __call__(self, shm_key: str, shape: tuple, dtype):
        """
        向推理服务器请求推理结果
        参数:
            shm_key: 共享内存 Key (如 "shm_1")
            shape: 图像形状 (h, w, c)
            dtype: 数据类型 (如 np.uint8)
        """
        with self.lock:
            if not self.sock:
                if not self._connect():
                    return None
            try:
                send_pickle(self.sock, {
                    'shm_key': shm_key,
                    'shape': shape,
                    'dtype': np.dtype(dtype).name
                })
                result = recv_pickle(self.sock)
                return result
            except Exception as e:
                print(f"[InferClient] {self.model_name} 请求异常: {e}")
                if self.sock:
                    self.sock.close()
                self.sock = None
                return None

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
