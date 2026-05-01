import signal
import threading


class ShutdownController:
    """Small signal-safe shutdown helper for standalone test scripts."""

    def __init__(self, name):
        self.name = name
        self.event = threading.Event()
        self._started = threading.Event()
        self._callbacks = []

    def install(self):
        signal.signal(signal.SIGINT, self.request)
        signal.signal(signal.SIGTERM, self.request)
        return self

    def request(self, signum=None, frame=None):
        del frame
        if self._started.is_set():
            return
        self._started.set()
        if signum is None:
            print(f"\n[{self.name}] Shutdown requested.")
        else:
            print(f"\n[{self.name}] Signal {signum} received, shutting down.")
        self.event.set()
        for callback in list(self._callbacks):
            try:
                callback()
            except Exception as exc:
                print(f"[{self.name}] Shutdown callback failed: {exc}")

    def add_callback(self, callback):
        self._callbacks.append(callback)
        return self

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout):
        return self.event.wait(timeout=timeout)
