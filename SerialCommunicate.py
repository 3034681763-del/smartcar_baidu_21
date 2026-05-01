import glob
import os
import struct
import time
from queue import Empty, Queue
from threading import Thread

import serial


class SerialCommunicate:
    def __init__(self, physical_path, baudrate, action_done_event=None):
        self.ser = None
        self.last_response = {}
        self._running = False
        self.thread_recv = None
        self.action_done_event = action_done_event
        self.last_action_done = None
        serial_device = self.find_serial_by_path(physical_path)
        if serial_device:
            print(f"[UART] Connected device: {serial_device}")
            try:
                self.ser = serial.Serial(serial_device, baudrate, timeout=0.5)
                if not self.ser.isOpen():
                    self.ser.open()
            except Exception as exc:
                print(f"[UART] Failed to open serial port: {exc}")
                self.ser = None
                return
        else:
            print(f"[UART] Device not found for path: {physical_path}")
            self.ser = None
            return

        self._running = True
        self.thread_recv = Thread(target=self.get_data_thread, args=(), daemon=True)
        self.thread_recv.start()

    def find_serial_by_path(self, physical_path):
        import pyudev

        if not physical_path:
            return None

        if physical_path.startswith("/dev/") and os.path.exists(physical_path):
            return physical_path

        by_path_dir = "/dev/serial/by-path"
        if os.path.isdir(by_path_dir):
            for link_name in sorted(os.listdir(by_path_dir)):
                if physical_path in link_name:
                    resolved_path = os.path.realpath(os.path.join(by_path_dir, link_name))
                    if os.path.exists(resolved_path):
                        print(f"[UART] Matched by-path entry: {link_name} -> {resolved_path}")
                        return resolved_path

        context = pyudev.Context()
        for subsystem in ("usb-serial", "tty"):
            for device in context.list_devices(subsystem=subsystem):
                device_node = getattr(device, "device_node", None)
                if not device_node:
                    continue

                path_candidates = [
                    getattr(device, "device_path", ""),
                    getattr(device, "sys_path", ""),
                ]
                if any(path and physical_path in path for path in path_candidates):
                    return device_node

                links = getattr(device, "device_links", [])
                if any(physical_path in link for link in links):
                    return device_node

        for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
            for device_node in sorted(glob.glob(pattern)):
                if os.path.exists(device_node):
                    print(f"[UART] Fallback serial candidate: {device_node}")
                    return device_node

        return None

    @staticmethod
    def packet_to_hex(packet):
        return " ".join(f"{byte:02X}" for byte in packet)

    def _read_exact(self, size):
        chunks = []
        total = 0
        while total < size:
            chunk = self.ser.read(size - total)
            if not chunk:
                return None
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks)

    def _mark_action_done(self, flag):
        self.last_action_done = {"flag": int(flag), "timestamp": time.time()}
        if self.action_done_event is not None:
            self.action_done_event.set()

    def get_data_thread(self):
        while self._running:
            ser = self.ser
            if ser is None:
                break

            try:
                if ser.in_waiting <= 0:
                    time.sleep(0.005)
                    continue
            except Exception as exc:
                if self._running:
                    print(f"[UART] Receive error: {exc}")
                break

            try:
                start = self._read_exact(1)
                if not start or struct.unpack("=B", start)[0] != 0x42:
                    continue

                address = self._read_exact(1)
                len_byte = self._read_exact(1)
                if not address or not len_byte:
                    continue

                cmd = struct.unpack("=B", address)[0]
                len_value = struct.unpack("=B", len_byte)[0]

                if cmd == 0x05:
                    payload = self._read_exact(len_value)
                    tail = self._read_exact(1)
                    if payload is None or not tail:
                        continue
                    if struct.unpack("=B", tail)[0] != 0x3C:
                        continue
                    if len_value == 1 and payload[0] == 0x05:
                        self._mark_action_done(payload[0])
                    else:
                        print("[UART] Invalid action-done payload, frame dropped")
                    continue

                tmp_data = [start, address, len_byte]
                if not len_byte:
                    continue
                frame_len = len_value
                remaining = frame_len - 3
                if remaining <= 0:
                    continue
                for _ in range(remaining):
                    byte = self._read_exact(1)
                    if not byte:
                        tmp_data = []
                        break
                    tmp_data.append(byte)
                if not tmp_data:
                    continue

                if struct.unpack("=B", tmp_data[frame_len - 1])[0] != 0x3C:
                    continue

                data_field = tmp_data[3:frame_len - 1]

                if cmd == 0x01:
                    if len(data_field) == 18:
                        byte_data = b"".join(data_field)
                        facheflag, cheatflag, dist_sensorl, dist_sensorr, world_y, world_z_angle = struct.unpack(
                            "=BBffff", bytes(byte_data)
                        )
                        if cheatflag == 123:
                            dist_sensorl = int(dist_sensorl)
                            dist_sensorr = int(dist_sensorr)
                            world_y = int(world_y)
                            world_z_angle = int(world_z_angle)

                        self.last_response = {
                            "Fache_flag": facheflag,
                            "Cheat_flag": cheatflag,
                            "dist_sensorL": dist_sensorl,
                            "dist_sensorR": dist_sensorr,
                            "world_y": world_y,
                            "world_z_angle": world_z_angle,
                        }
                    else:
                        print("[UART] Invalid payload length for cmd 0x01, frame dropped")
                else:
                    print(f"[UART] Unknown upstream cmd: {cmd}")
            except Exception as exc:
                if self._running:
                    print(f"[UART] Receive error: {exc}")
                break

    def send_motion_mode(self, mode, deviation=None, speed_x=None, pos_x=None, pos_y=None, z_angle=None):
        if self.ser is None:
            return

        packet = None
        desc = ""
        if mode == 0:
            if deviation is None or speed_x is None:
                return
            packet = struct.pack("=BBBB2fB", 0x42, 0x11, 13, 0, float(speed_x), float(deviation), 0x3C)
            desc = f"mode=0 speed_x={float(speed_x):.2f} deviation={float(deviation):.2f}"
        elif mode == 1:
            if pos_x is None or pos_y is None or z_angle is None:
                return
            packet = struct.pack("=BBBB3fB", 0x42, 0x11, 17, 1, float(pos_x), float(pos_y), float(z_angle), 0x3C)
            desc = f"mode=1 pos_x={float(pos_x):.2f} pos_y={float(pos_y):.2f} z_angle={float(z_angle):.2f}"

        if packet is None:
            return

        self.ser.write(packet)
        print(f"[UART TX] {desc}")
        print(f"[UART TX HEX] {self.packet_to_hex(packet)}")

    def send_arm_motion(self, mot0, mot1, mot2, mot3, suck, light):
        if self.ser is None:
            return

        packet = struct.pack(
            "=3b4h3b",
            0x42,
            0x12,
            14,
            int(mot0),
            int(mot1),
            int(mot2),
            int(mot3),
            int(suck),
            int(light),
            0x3C,
        )
        self.ser.write(packet)
        print(
            "[UART TX] "
            f"arm mot0={mot0} mot1={mot1} mot2={mot2} mot3={mot3} suck={suck} light={light}"
        )
        print(f"[UART TX HEX] {self.packet_to_hex(packet)}")

    def send_mode_flag(self, flag):
        if self.ser is None:
            return

        packet = struct.pack("=BBBBB", 0x42, 0x13, 0x05, int(flag), 0x3C)
        self.ser.write(packet)
        print(f"[UART TX] sysmode flag={flag}")
        print(f"[UART TX HEX] {self.packet_to_hex(packet)}")

    def send_align_done(self):
        if self.ser is None:
            return

        packet = bytes([0x42, 0x06, 0x00, 0x3C])
        for repeat_index in range(10):
            self.ser.write(packet)
            print(f"[UART TX] align done {repeat_index + 1}/10")
            print(f"[UART TX HEX] {self.packet_to_hex(packet)}")
            time.sleep(0.02)

    def close(self):
        self._running = False
        thread = self.thread_recv
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

        ser = self.ser
        self.ser = None
        if ser and ser.isOpen():
            try:
                ser.close()
            except Exception:
                pass


class SerialServer:
    def __init__(
        self,
        serial_path="1-2.1:1.0",
        baudrate=115200,
        request_queue=None,
        publish_queue=None,
        action_done_event=None,
    ):
        self.serial_comm = SerialCommunicate(serial_path, baudrate, action_done_event=action_done_event)
        self.flag_exit = False
        self.request_queue = request_queue or Queue()
        self.publish_queue = publish_queue or Queue()
        self.last_pub_action_done_ts = 0.0

    def process_rep_requests(self, timeout=0.1):
        while not self.flag_exit:
            try:
                req = self.request_queue.get(timeout=timeout)
            except Empty:
                continue
            except Exception as exc:
                print(f"[SerialServer] Queue read error: {exc}")
                continue

            try:
                cmd_type = req.get("cmd")
                if cmd_type == "Motion":
                    mode = req.get("mode")
                    if mode == 0:
                        self.serial_comm.send_motion_mode(
                            0,
                            deviation=req["deviation"],
                            speed_x=req["speed_x"],
                        )
                    elif mode == 1:
                        self.serial_comm.send_motion_mode(
                            1,
                            pos_x=req["pos_x"],
                            pos_y=req["pos_y"],
                            z_angle=req["z_angle"],
                        )
                elif cmd_type == "Arm":
                    self.serial_comm.send_arm_motion(
                        req["mot0"],
                        req["mot1"],
                        req["mot2"],
                        req["mot3"],
                        req.get("suck", 0),
                        req.get("light", 0),
                    )
                elif cmd_type == "SysMode":
                    self.serial_comm.send_mode_flag(req["flag"])
                elif cmd_type == "AlignDone":
                    self.serial_comm.send_align_done()
            except Exception as exc:
                print(f"[SerialServer] Request error: {exc}")

    def pub_loop(self):
        while not self.flag_exit:
            data = self.serial_comm.last_response
            if data:
                self.publish_queue.put({"cmd": "PushResp", "data": data})
            action_done = self.serial_comm.last_action_done
            if action_done and action_done["timestamp"] > self.last_pub_action_done_ts:
                self.publish_queue.put({"cmd": "ActionDone", "data": {"flag": action_done["flag"]}})
                self.last_pub_action_done_ts = action_done["timestamp"]
            time.sleep(0.02)

    def run(self, stop_event=None):
        t1 = Thread(target=self.process_rep_requests, daemon=True)
        t2 = Thread(target=self.pub_loop, daemon=True)
        t1.start()
        t2.start()

        if stop_event:
            stop_event.wait()
            print("[SerialServer] Stop signal received, closing serial service")
            self.close()

    def close(self):
        self.flag_exit = True
        self.serial_comm.close()


def serial_server_process(
    stop_event,
    physical_path="1-2.1:1.0",
    baudrate=115200,
    request_queue=None,
    publish_queue=None,
    action_done_event=None,
):
    print("[SerialServer] Hardware communication subprocess started")
    server = SerialServer(
        serial_path=physical_path,
        baudrate=baudrate,
        request_queue=request_queue,
        publish_queue=publish_queue,
        action_done_event=action_done_event,
    )
    server.run(stop_event)
