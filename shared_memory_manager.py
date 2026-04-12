# --------------- shared_memory_manager.py ---------------
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.resource_tracker import unregister
import struct

class SharedMemoryManager:
    def __init__(self):
        # 使用字典维护所有共享内存块的引用
        self.memory_blocks = {}  # {name: SharedMemory}
        
    def create_block(self, key: str, size: int) -> None:
        """创建或附加到现有的共享内存块"""
        try:
            # 尝试创建新内存块
            shm = SharedMemory(name=key, create=True, size=size)
        except FileExistsError:
            # 附加到已有内存块
            shm = SharedMemory(name=key)
        
        # 解除资源追踪（避免程序退出时Linux/Windows内核自动过早回收，交由代码手动统一释放）
        unregister(shm._name, 'shared_memory')
        
        self.memory_blocks[key] = shm

    def write_image(self, key: str, img: np.ndarray) -> None:
        """将图像写入大块共享内存 (通常用于 shm_0, shm_1)"""
        if key not in self.memory_blocks:
            raise KeyError(f"Shared memory block {key} not exists")
            
        shm = self.memory_blocks[key]
        if img.nbytes > shm.size:
            raise ValueError(
                f"Image size {img.nbytes} exceeds shared memory block size {shm.size}"
            )
        
        # 将图像数据覆盖到共享内存缓冲区
        shm.buf[:img.nbytes] = img.tobytes()

    def read_image(self, key: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """从大块共享内存读取图像重建 numpy 数组"""
        if key not in self.memory_blocks:
            raise KeyError(f"Shared memory block {key} not exists")
            
        shm = self.memory_blocks[key]
        
        # 从缓冲区重建numpy数组
        arr = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=shm.buf,
            order='C'
        )
        return np.copy(arr)  # 返回拷贝以避免异步读写带来的内存竞争/撕裂

    def write_floats(self, key: str, values: tuple) -> None:
        """打包写入一组 float 浮点数 (专用于巡线特征 shm_lane等8字节短内存)"""
        if key not in self.memory_blocks:
            raise KeyError(f"Shared memory block {key} not exists")
        shm = self.memory_blocks[key]
        
        # 将 float 打包为字节流，='ff' 代表两个float
        fmt = '=' + 'f' * len(values)
        data_bytes = struct.pack(fmt, *values)
        
        if len(data_bytes) > shm.size:
            raise ValueError("Values size exceeds shared memory capacity")
            
        shm.buf[:len(data_bytes)] = data_bytes

    def read_floats(self, key: str, num_floats: int) -> tuple:
        """读取被打包的浮点数"""
        if key not in self.memory_blocks:
            raise KeyError(f"Shared memory block {key} not exists")
        shm = self.memory_blocks[key]
        
        fmt = '=' + 'f' * num_floats
        byte_size = struct.calcsize(fmt)
        data_bytes = shm.buf[:byte_size]
        
        return struct.unpack(fmt, data_bytes)

    def release_block(self, key: str) -> None:
        """安全释放指定名称的共享内存资源"""
        if key in self.memory_blocks:
            shm = self.memory_blocks.pop(key)
            shm.close()
            try:
                shm.unlink()  # 彻底销毁内核对应的内存块
            except FileNotFoundError:
                pass

    def release_all(self):
        """释放所有管控的内存块"""
        for key in list(self.memory_blocks.keys()):
            self.release_block(key)

    def __del__(self):
        """析构时自动释放"""
        self.release_all()
