import zmq
import json
import pickle
from typing import Any, Optional
import os

"""
ZMQComm: 提供极简的微服务通信模块。
支持的模式: 'push', 'pull', 'pub', 'sub', 'req', 'rep'
地址示例: 'tcp://127.0.0.1:5555' (跨设备), 'ipc:///tmp/zmq.ipc' (跨进程), 'inproc://test' (跨线程)
"""
class ZMQComm:
    _shared_context: Optional[zmq.Context] = None

    def __init__(self, mode: str, address: str, buffer_size: int = 16, context: Optional[zmq.Context] = None):
        if context is None:
            if ZMQComm._shared_context is None:
                ZMQComm._shared_context = zmq.Context.instance()
            self.context = ZMQComm._shared_context
        else:
            self.context = context

        self.mode = mode.lower()
        self.address = address
        self.buffer_size = buffer_size
        self.socket = self._create_socket()

    def _handle_ipc_address(self):
        if self.address.startswith("ipc://"):
            path = self.address[6:]
            if os.path.exists(path):
                os.remove(path)

    def _create_socket(self):
        socket_type_map = {
            'push': zmq.PUSH,
            'pull': zmq.PULL,
            'pub':  zmq.PUB,
            'sub':  zmq.SUB,
            'req':  zmq.REQ,
            'rep':  zmq.REP,
        }
        socket_type = socket_type_map.get(self.mode)
        if socket_type is None:
            raise ValueError("不支持的 mode")
        
        socket = self.context.socket(socket_type)

        if self.mode == 'sub':
            socket.setsockopt_string(zmq.SUBSCRIBE, "")

        if self.mode in ['sub', 'pub']:
            socket.setsockopt(zmq.SNDHWM, 1)  # 仅保留最新的一条，丢弃旧消息防堵塞
            socket.setsockopt(zmq.RCVHWM, 1)  
        elif self.mode in ['req', 'rep']:
            socket.setsockopt(zmq.SNDHWM, 0)
            socket.setsockopt(zmq.RCVHWM, 0)

        if self.mode in ['rep', 'pub', 'push']:
            if self.address.startswith("ipc://"):
                self._handle_ipc_address()
            socket.bind(self.address)
        elif self.mode in ['req', 'sub', 'pull']:
            socket.connect(self.address)
        
        return socket

    def send(self, data: Any):
        if isinstance(data, str):
            message = data.encode('utf-8')
        elif isinstance(data, dict):
            message = json.dumps(data).encode('utf-8')
        elif isinstance(data, bytes):
            message = data
        else:
            message = pickle.dumps(data)

        self.socket.send(message, zmq.NOBLOCK)

    def receive(self) -> Any:
        message = self.socket.recv()
        try:
            return message.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return json.loads(message.decode('utf-8'))
            except json.JSONDecodeError:
                return pickle.loads(message)

    def close(self):
        self.socket.close()

    def subscribe(self, filter: str = ""):
        if self.mode == "sub":
            self.socket.setsockopt_string(zmq.SUBSCRIBE, filter)

    def unsubscribe(self, filter: str = ""):
        if self.mode == "sub":
            self.socket.setsockopt_string(zmq.UNSUBSCRIBE, filter)
