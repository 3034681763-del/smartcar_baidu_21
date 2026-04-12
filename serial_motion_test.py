import argparse
import struct
import sys
import time


def build_motion_packet(speed_x: float, deviation: float) -> bytes:
    return struct.pack(
        "=BBBB2fB",
        0x42,
        0x11,
        13,
        0,
        float(speed_x),
        float(deviation),
        0x3C,
    )


def format_hex(packet: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in packet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the mode-0 motion packet used by the lower controller."
    )
    parser.add_argument(
        "--speed-x",
        type=float,
        default=-0.28,
        help="Forward speed. Default matches the current lane-following code.",
    )
    parser.add_argument(
        "--deviation",
        type=float,
        default=0.0,
        help="Lane deviation. Default 0 means centered lane following.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="",
        help="Serial port to send to, for example COM3 or /dev/ttyUSB0. If omitted, only print the packet.",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="How many packets to send. Use 0 to send continuously until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Seconds between packets. Default 0.05s is 20Hz.",
    )
    return parser.parse_args()


def send_packets(args: argparse.Namespace, packet: bytes) -> int:
    try:
        import serial
    except ImportError:
        print("pyserial 未安装，当前只能打印数据包，无法真实发送串口。")
        print("可安装：pip install pyserial")
        return 1

    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.5) as ser:
            if args.count == 0:
                index = 0
                print("持续发送中，按 Ctrl+C 停止。")
                while True:
                    index += 1
                    ser.write(packet)
                    ser.flush()
                    print(
                        f"[{index}] sent -> {format_hex(packet)} "
                        f"(speed_x={args.speed_x}, deviation={args.deviation})"
                    )
                    time.sleep(args.interval)
            else:
                for index in range(args.count):
                    ser.write(packet)
                    ser.flush()
                    print(
                        f"[{index + 1}/{args.count}] sent -> {format_hex(packet)} "
                        f"(speed_x={args.speed_x}, deviation={args.deviation})"
                    )
                    if index != args.count - 1:
                        time.sleep(args.interval)
    except KeyboardInterrupt:
        print("用户中断，已停止发送。")
    except Exception as exc:
        print(f"串口发送失败: {exc}")
        return 1

    return 0


def main() -> int:
    args = parse_args()
    packet = build_motion_packet(args.speed_x, args.deviation)

    print("Motion mode-0 packet ready")
    print(f"speed_x   : {args.speed_x}")
    print(f"deviation : {args.deviation}")
    print(f"length    : {len(packet)}")
    print(f"bytes(hex): {format_hex(packet)}")

    unpacked = struct.unpack("=BBBB2fB", packet)
    print(f"unpacked  : {unpacked}")

    if not args.port:
        print("未指定 --port，仅做打包测试，没有真实发送。")
        print("真实发送示例: python serial_motion_test.py --port COM3")
        return 0

    print(f"串口发送目标: {args.port} @ {args.baudrate}")
    return send_packets(args, packet)


if __name__ == "__main__":
    sys.exit(main())
