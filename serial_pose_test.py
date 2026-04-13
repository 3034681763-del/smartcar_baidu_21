import argparse
import struct
import sys
import time


def build_pose_packet(pos_x: float, pos_y: float, z_angle: float) -> bytes:
    return struct.pack(
        "=BBBB3fB",
        0x42,
        0x11,
        17,
        1,
        float(pos_x),
        float(pos_y),
        float(z_angle),
        0x3C,
    )


def format_hex(packet: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in packet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the mode-1 pose packet used by the lower controller."
    )
    parser.add_argument("--pos-x", type=float, default=0.0, help="Target x displacement.")
    parser.add_argument("--pos-y", type=float, default=0.0, help="Target y displacement.")
    parser.add_argument("--z-angle", type=float, default=0.0, help="Target yaw angle.")
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
        default=10,
        help="How many packets to send. Use 0 to send continuously until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between packets. Default 0.1s is 10Hz.",
    )
    return parser.parse_args()


def send_packets(args: argparse.Namespace, packet: bytes) -> int:
    try:
        import serial
    except ImportError:
        print("pyserial is not installed, so the script can only print the packet.")
        print("Install it with: pip install pyserial")
        return 1

    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.5) as ser:
            if args.count == 0:
                index = 0
                print("Sending continuously, press Ctrl+C to stop.")
                while True:
                    index += 1
                    ser.write(packet)
                    ser.flush()
                    print(
                        f"[{index}] sent -> {format_hex(packet)} "
                        f"(pos_x={args.pos_x}, pos_y={args.pos_y}, z_angle={args.z_angle})"
                    )
                    time.sleep(args.interval)
            else:
                for index in range(args.count):
                    ser.write(packet)
                    ser.flush()
                    print(
                        f"[{index + 1}/{args.count}] sent -> {format_hex(packet)} "
                        f"(pos_x={args.pos_x}, pos_y={args.pos_y}, z_angle={args.z_angle})"
                    )
                    if index != args.count - 1:
                        time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exc:
        print(f"Serial send failed: {exc}")
        return 1

    return 0


def main() -> int:
    args = parse_args()
    packet = build_pose_packet(args.pos_x, args.pos_y, args.z_angle)

    print("Motion mode-1 packet ready")
    print(f"pos_x     : {args.pos_x}")
    print(f"pos_y     : {args.pos_y}")
    print(f"z_angle   : {args.z_angle}")
    print(f"length    : {len(packet)}")
    print(f"bytes(hex): {format_hex(packet)}")

    unpacked = struct.unpack("=BBBB3fB", packet)
    print(f"unpacked  : {unpacked}")

    if not args.port:
        print("No --port was provided, so this was only a packet build test.")
        print("Example real send: python serial_pose_test.py --port COM3 --pos-x 0 --pos-y 100 --z-angle 0")
        return 0

    print(f"Serial target: {args.port} @ {args.baudrate}")
    return send_packets(args, packet)


if __name__ == "__main__":
    sys.exit(main())
