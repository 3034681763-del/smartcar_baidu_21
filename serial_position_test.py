import argparse
import struct
import sys
import time


def build_position_packet(pos_x: float, pos_y: float, z_angle: float) -> bytes:
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
        description="Test the mode-1 position packet used by the lower controller."
    )
    parser.add_argument("--pos-x", type=float, default=0.0, help="Target x displacement.")
    parser.add_argument("--pos-y", type=float, default=0.0, help="Target y displacement.")
    parser.add_argument("--z-angle", type=float, default=15.0, help="Target yaw angle.")
    parser.add_argument(
        "--port",
        type=str,
        default="",
        help="Serial port to send to, for example /dev/ttyUSB0 or COM3.",
    )
    parser.add_argument(
        "--physical-path",
        type=str,
        default="",
        help="USB physical path used by SerialCommunicate, for example 1-2.1:1.0.",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate.")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many packets to send. Use 0 to send continuously until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between packets.",
    )
    return parser.parse_args()


def send_via_port(args: argparse.Namespace, packet: bytes) -> int:
    try:
        import serial
    except ImportError:
        print("pyserial is not installed, so --port mode cannot send serial data.")
        print("Install with: pip install pyserial")
        return 1

    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.5) as ser:
            return send_loop(args, writer=lambda: ser.write(packet))
    except KeyboardInterrupt:
        print("Stopped by user.")
        return 0
    except Exception as exc:
        print(f"Serial send failed: {exc}")
        return 1


def send_via_project_uart(args: argparse.Namespace) -> int:
    try:
        from SerialCommunicate import SerialCommunicate
    except ImportError as exc:
        print(f"Could not import project serial module: {exc}")
        return 1

    uart = None
    try:
        uart = SerialCommunicate(args.physical_path, args.baudrate)
        if uart.ser is None:
            print("Project serial init failed. Check physical path and permissions.")
            return 1

        def writer() -> None:
            uart.send_motion_mode(
                1,
                pos_x=args.pos_x,
                pos_y=args.pos_y,
                z_angle=args.z_angle,
            )

        return send_loop(args, writer=writer)
    except KeyboardInterrupt:
        print("Stopped by user.")
        return 0
    except Exception as exc:
        print(f"Project UART send failed: {exc}")
        return 1
    finally:
        if uart is not None:
            uart.close()


def send_loop(args: argparse.Namespace, writer) -> int:
    if args.count == 0:
        index = 0
        print("Sending continuously. Press Ctrl+C to stop.")
        while True:
            index += 1
            writer()
            print_status(args, index)
            time.sleep(args.interval)
    else:
        for index in range(1, args.count + 1):
            writer()
            print_status(args, index, args.count)
            if index != args.count:
                time.sleep(args.interval)
    return 0


def print_status(args: argparse.Namespace, index: int, total=None) -> None:
    prefix = f"[{index}]" if total is None else f"[{index}/{total}]"
    print(f"{prefix} sent -> pos_x={args.pos_x}, pos_y={args.pos_y}, z_angle={args.z_angle}")


def main() -> int:
    args = parse_args()
    packet = build_position_packet(args.pos_x, args.pos_y, args.z_angle)

    print("Motion mode-1 packet ready")
    print(f"pos_x     : {args.pos_x}")
    print(f"pos_y     : {args.pos_y}")
    print(f"z_angle   : {args.z_angle}")
    print(f"length    : {len(packet)}")
    print(f"bytes(hex): {format_hex(packet)}")
    print(f"unpacked  : {struct.unpack('=BBBB3fB', packet)}")

    if args.port and args.physical_path:
        print("Use either --port or --physical-path, not both.")
        return 1

    if args.port:
        print(f"Sending to serial port: {args.port} @ {args.baudrate}")
        return send_via_port(args, packet)

    if args.physical_path:
        print(f"Searching serial device by physical path: {args.physical_path} @ {args.baudrate}")
        return send_via_project_uart(args)

    print("No --port or --physical-path was provided, so this run only validates packet packing.")
    print("Example 1: python3 serial_position_test.py --port /dev/ttyUSB0 --pos-x 0 --pos-y 0 --z-angle 30")
    print("Example 2: python3 serial_position_test.py --physical-path 1-2.1:1.0 --pos-x 0 --pos-y 0 --z-angle 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
