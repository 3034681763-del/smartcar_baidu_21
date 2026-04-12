import struct
import time
from threading import Thread

import serial

import Zmq_mod


class SerialCommunicate:
    def __init__(self, physical_path, baudrate):
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

        self.last_response = {}
        self.thread_recv = Thread(target=self.get_data_thread, args=(), daemon=True)
        self.thread_recv.start()

    def find_serial_by_path(self, physical_path):
        import pyudev

        context = pyudev.Context()
        for device in context.list_devices(subsystem="usb-serial"):
            path = device.device_path.rsplit("/", 1)[0]
            if path.endswith(physical_path):
                return "/dev/" + device.device_path.rsplit("/", 1)[1]
        return None

    @staticmethod
    def packet_to_hex(packet):
        return " ".join(f"{byte:02X}" for byte in packet)

    def get_data_thread(self):
        while True:
            if self.ser.in_waiting <= 0:
                time.sleep(0.005)
                continue

            try:
                start = self.ser.read(1)
                if not start or struct.unpack("=b", start)[0] != 0x42:
                    continue

                tmp_data = [start]
                address = self.ser.read(1)
                tmp_data.append(address)

                len_byte = self.ser.read(1)
                if not len_byte:
                    continue
                tmp_data.append(len_byte)

                frame_len = struct.unpack("=b", len_byte)[0]
                remaining = frame_len - 3
                for _ in range(remaining):
                    tmp_data.append(self.ser.read(1))

                if struct.unpack("=b", tmp_data[frame_len - 1])[0] != 0x3C:
                    continue

                data_field = tmp_data[3:frame_len - 1]
                cmd = struct.unpack("=b", tmp_data[1])[0]

                if cmd == 0x01 and len(data_field) == 18:
                    byte_data = b"".join(data_field)
                    facheflag, cheatflag, dist_sensorl, dist_sensorr, world_y, world_z_angle = struct.unpack(
                        "=BBffff", bytes(byte_data)
                    )
                    self.last_response = {
                        "Fache_flag": facheflag,
                        "Cheat_flag": cheatflag,
                        "dist_sensorL": dist_sensorl,
                        "dist_sensorR": dist_sensorr,
                        "world_y": world_y,
                        "world_z_angle": world_z_angle,
                    }
            except Exception as exc:
                print(f"[UART] Receive error: {exc}")

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

    def close(self):
        if self.ser and self.ser.isOpen():
            self.ser.close()


class SerialServer:
    def __init__(self, serial_path="1-2.1:1.0", baudrate=115200):
        self.serial_comm = SerialCommunicate(serial_path, baudrate)
        self.zmq_req = Zmq_mod.ZMQComm(mode="rep", address="ipc:///tmp/MainWithUart.ipc")
        self.zmq_pub = Zmq_mod.ZMQComm(mode="pub", address="ipc:///tmp/SubUartInfo.ipc")
        self.flag_exit = False

    def process_rep_requests(self):
        while not self.flag_exit:
            try:
                req = self.zmq_req.receive()
                if not req:
                    continue

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

                self.zmq_req.send({"status": "ok"})
            except Exception as exc:
                print(f"[SerialServer] Request error: {exc}")

    def pub_loop(self):
        while not self.flag_exit:
            data = self.serial_comm.last_response
            if data:
                self.zmq_pub.send({"cmd": "PushResp", "data": data})
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
        self.zmq_req.close()
        self.zmq_pub.close()
        self.serial_comm.close()


def serial_server_process(stop_event, physical_path="1-2.1:1.0"):
    print("[SerialServer] Hardware communication subprocess started")
    server = SerialServer(serial_path=physical_path)
    server.run(stop_event)
