#!/usr/bin/env python3
import argparse
import time

from SerialCommunicate import SerialCommunicate


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone serial receive test with optional one-shot send."
    )
    parser.add_argument(
        "--physical-path",
        default="2.2:1.0",
        help="Serial physical path, for example 2.2:1.0 or /dev/ttyUSB0",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--send-pose",
        action="store_true",
        help="Send one mode-1 pose command after connection",
    )
    parser.add_argument("--pos-x", type=float, default=0.0, help="Pose test pos_x")
    parser.add_argument("--pos-y", type=float, default=0.0, help="Pose test pos_y")
    parser.add_argument("--z-angle", type=float, default=0.0, help="Pose test z_angle")
    return parser


def main():
    args = build_parser().parse_args()
    uart = SerialCommunicate(args.physical_path, args.baudrate)
    if uart.ser is None:
        print("[serial_test] Serial init failed.")
        return 1

    print("=" * 60)
    print("Serial Test")
    print(f"Physical path: {args.physical_path}")
    print(f"Baudrate:      {args.baudrate}")
    print("Quit:          Ctrl+C")
    print("=" * 60)

    try:
        if args.send_pose:
            uart.send_motion_mode(
                1,
                pos_x=args.pos_x,
                pos_y=args.pos_y,
                z_angle=args.z_angle,
            )

        last_payload = None
        while True:
            payload = uart.last_response
            if payload and payload != last_payload:
                print(f"[serial_test] RX -> {payload}")
                last_payload = dict(payload)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[serial_test] Stopped by user.")
        return 0
    finally:
        uart.close()


if __name__ == "__main__":
    raise SystemExit(main())
