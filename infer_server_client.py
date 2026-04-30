# !/usr/bin/env python3
import os
import pickle
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from model_path_config import get_task_model_path

DEFAULT_PADDLE_JETSON_PATH = "/home/jetson/workspace/code_vehicle_wbt/paddle_jetson"


model_configs = [
    {
        "name": "task",
        "infer_type": "YoloeInfer",
        "params": [get_task_model_path()],
        "port": 5020,
        "img_size": [640, 640],
    },
    {
        "name": "word",
        "infer_type": "OCRReco",
        "params": ["fenge_mod"],
        "port": 5010,
        "img_size": [640, 640],
    },
]


def ensure_paddle_jetson_importable():
    configured_path = os.environ.get("PADDLE_JETSON_PATH", DEFAULT_PADDLE_JETSON_PATH)
    if not configured_path:
        return

    normalized_path = os.path.abspath(configured_path)
    search_path = (
        os.path.dirname(normalized_path)
        if os.path.basename(normalized_path) == "paddle_jetson"
        else normalized_path
    )

    if search_path not in sys.path:
        sys.path.insert(0, search_path)

    print(f"[InferServer] paddle_jetson search path: {search_path}")


def send_pickle(conn: socket.socket, obj):
    try:
        file = conn.makefile("wb")
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
    except Exception as exc:
        raise RuntimeError(f"send_pickle failed: {exc}") from exc


def recv_pickle(conn: socket.socket):
    try:
        file = conn.makefile("rb")
        return pickle.load(file)
    except Exception as exc:
        raise RuntimeError(f"recv_pickle failed: {exc}") from exc


class ModelManager:
    def __init__(self, cfg):
        ensure_paddle_jetson_importable()
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            from paddle_jetson import OCRReco, YoloeInfer
        finally:
            sys.argv = original_argv

        model_map = {
            "OCRReco": OCRReco,
            "YoloeInfer": YoloeInfer,
        }
        infer_cls = model_map[cfg["infer_type"]]
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]
        try:
            self.model = infer_cls(*cfg["params"])
        finally:
            sys.argv = original_argv

        if "img_size" in cfg:
            h, w = cfg["img_size"]
            blank_img = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(2):
                try:
                    self.model.predict(blank_img)
                except Exception as exc:
                    print(f"[ModelManager] Warmup failed for {cfg['name']}: {exc}")

        print(f"[ModelManager] {cfg['name']} is ready")

    def predict(self, img):
        return self.model.predict(img)

    def close(self):
        close_fn = getattr(self.model, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


class ModelProcessServer:
    def __init__(self, cfg, shm_manager):
        self.cfg = cfg
        self.shm_manager = shm_manager
        self.model_name = cfg["name"]
        self.model_mgr = ModelManager(cfg)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.bind(("0.0.0.0", cfg["port"]))
        self.sock.listen(8)
        self.running = True
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        print(f"[{self.model_name} Server] Listening on {cfg['port']}")

    def start(self, stop_event):
        self.sock.settimeout(0.5)
        try:
            while not stop_event.is_set():
                try:
                    conn, addr = self.sock.accept()
                except socket.timeout:
                    continue
                print(f"[{self.model_name} Server] Connection from {addr}")
                self.thread_pool.submit(self.handle_connection, conn)
        finally:
            self.running = False
            self.sock.close()
            self.thread_pool.shutdown(wait=True)
            self.model_mgr.close()
            print(f"[{self.model_name} Server] Shutdown complete")

    def handle_connection(self, conn):
        with conn:
            while self.running:
                try:
                    req = recv_pickle(conn)
                    if isinstance(req, dict) and req.get("handshake", False):
                        send_pickle(conn, {"ready": True})
                        continue

                    img = self.shm_manager.read_image(
                        req["shm_key"],
                        tuple(req["shape"]),
                        np.dtype(req["dtype"]),
                    )
                    result = self.model_mgr.predict(img)
                    send_pickle(conn, result)
                except Exception as exc:
                    print(f"[{self.model_name} Server] Request failed: {exc}")
                    break


def serve_model_process(stop_event, cfg, shm_manager):
    server = ModelProcessServer(cfg, shm_manager)
    server.start(stop_event)


class InferClient1:
    def __init__(self, model_name: str, shm_manager, port: int, timeout=2):
        self.model_name = model_name
        self.shm_manager = shm_manager
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()
        self.reconnect_interval = 1
        self._block_until_server_ready()

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(self.timeout)
            self.sock.connect(("localhost", self.port))
            return True
        except Exception:
            self.sock = None
            return False

    def _block_until_server_ready(self):
        while True:
            if self._connect():
                try:
                    send_pickle(self.sock, {"handshake": True})
                    resp = recv_pickle(self.sock)
                    if isinstance(resp, dict) and resp.get("ready", False):
                        self.sock.close()
                        self.sock = None
                        print(f"[InferClient] {self.model_name} ready")
                        return
                except Exception:
                    if self.sock:
                        self.sock.close()
                    self.sock = None
            time.sleep(self.reconnect_interval)

    def __call__(self, shm_key: str, shape: tuple, dtype):
        with self.lock:
            if not self.sock and not self._connect():
                return None

            try:
                send_pickle(
                    self.sock,
                    {
                        "shm_key": shm_key,
                        "shape": shape,
                        "dtype": np.dtype(dtype).name,
                    },
                )
                return recv_pickle(self.sock)
            except Exception as exc:
                print(f"[InferClient] {self.model_name} request failed: {exc}")
                if self.sock:
                    self.sock.close()
                self.sock = None
                return None

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


def select_detections_by_label(detections, target_label="text", sort_by="y"):
    if not detections:
        return []

    filtered = [
        det for det in detections
        if getattr(det, "label", None) == target_label
    ]

    if sort_by == "x":
        filtered.sort(key=lambda det: float(det.center[0]))
    else:
        filtered.sort(key=lambda det: float(det.center[1]))
    return filtered


class OCRPipeline:
    def __init__(self, shm_manager, task_client, word_client, crop_key="shm_crop"):
        self.shm_manager = shm_manager
        self.task_client = task_client
        self.word_client = word_client
        self.crop_key = crop_key

    def read_texts(
        self,
        shm_key="shm_0",
        shape=(640, 640, 3),
        dtype=np.uint8,
        target_label="text",
        result_amount=1,
        sort_by="y",
        pad=12,
        retries=6,
    ):
        stable_texts = None
        stable_hits = 0

        for _ in range(retries):
            frame = self.shm_manager.read_image(shm_key, shape, np.dtype(dtype))
            detections = self.task_client(shm_key, shape, np.dtype(dtype))
            selected = select_detections_by_label(detections, target_label, sort_by)
            if not selected:
                time.sleep(0.05)
                continue

            texts = []
            for det in selected[:result_amount]:
                x1, y1, x2, y2 = map(int, det.bbox)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                self.shm_manager.write_image(self.crop_key, crop)
                text = self.word_client(self.crop_key, crop.shape, crop.dtype)
                if text:
                    texts.append(str(text).strip())

            if not texts:
                time.sleep(0.05)
                continue

            if texts == stable_texts:
                stable_hits += 1
            else:
                stable_texts = texts
                stable_hits = 1

            if stable_hits >= 2:
                return texts

            time.sleep(0.05)

        return stable_texts or []

    def read_text_items(
        self,
        shm_key="shm_0",
        shape=(640, 640, 3),
        dtype=np.uint8,
        target_label="text",
        result_amount=1,
        sort_by="y",
        pad=12,
        retries=6,
        require_count=None,
    ):
        last_items = []

        for _ in range(retries):
            frame = self.shm_manager.read_image(shm_key, shape, np.dtype(dtype))
            detections = self.task_client(shm_key, shape, np.dtype(dtype))
            selected = select_detections_by_label(detections, target_label, sort_by)
            if require_count is not None and len(selected) != int(require_count):
                time.sleep(0.05)
                continue
            if not selected:
                time.sleep(0.05)
                continue

            items = []
            for det in selected[:result_amount]:
                x1, y1, x2, y2 = map(int, det.bbox)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                self.shm_manager.write_image(self.crop_key, crop)
                text = self.word_client(self.crop_key, crop.shape, crop.dtype)
                if not text:
                    continue
                items.append(
                    {
                        "bbox": tuple(det.bbox),
                        "center": tuple(det.center),
                        "text": str(text).strip(),
                    }
                )

            if require_count is not None and len(items) != int(require_count):
                last_items = items
                time.sleep(0.05)
                continue
            if items:
                return items
            last_items = items
            time.sleep(0.05)

        return last_items
