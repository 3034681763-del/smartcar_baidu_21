import atexit
import multiprocessing
import os
import signal
import subprocess
import sys
from typing import Callable

import psutil


class ProcessManager:
    """Manage child processes whose first argument is a multiprocessing.Event."""

    def __init__(self, setup_signal_handlers=True):
        self.processes = []
        self._terminating = False
        self._setup_exit_handlers(setup_signal_handlers=setup_signal_handlers)

    def add_process(self, target: Callable, args: tuple = ()):
        stop_event = multiprocessing.Event()
        full_args = (stop_event,) + args
        process = multiprocessing.Process(target=target, args=full_args)
        self.processes.append((process, stop_event))

    def start_all(self):
        for process, _ in self.processes:
            process.start()

    def terminate_all(self, graceful_timeout=3.0):
        if self._terminating:
            return
        self._terminating = True

        for process, stop_event in self.processes:
            if process.is_alive():
                stop_event.set()

        for process, _ in self.processes:
            if not process.is_alive():
                continue
            process.join(timeout=graceful_timeout)
            if process.is_alive():
                print(f"[ProcessManager] Force terminating process pid={process.pid}")
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    print(f"[ProcessManager] Force killing process pid={process.pid}")
                    process.kill()
                    process.join(timeout=1.0)

    def _setup_exit_handlers(self, setup_signal_handlers=True):
        atexit.register(self.terminate_all)
        if setup_signal_handlers:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[ProcessManager] Caught signal {signum}, terminating subprocesses...")
        self.terminate_all()
        sys.exit(0)


def get_python_processes():
    python_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" in proc.info["name"].lower():
                cmdline = proc.info["cmdline"]
                if len(cmdline) > 1:
                    script_path = cmdline[1]
                    python_processes.append((proc.info["pid"], script_path))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return python_processes


def check_back_python(file_name):
    dir_file = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(dir_file, file_name)
    if not os.path.exists(file_path):
        raise Exception(f"File does not exist: {file_path}")

    py_processes = get_python_processes()
    found = False

    for pid, script_path in py_processes:
        del pid
        script_name = os.path.basename(script_path)
        if script_name == file_name:
            found = True
            break

    if not found:
        cmd = [sys.executable, file_path]
        subprocess.Popen(cmd)
        print(f"Started: {file_name} (output is shown in this terminal)")
    else:
        print(f"Already running: {file_name}")
