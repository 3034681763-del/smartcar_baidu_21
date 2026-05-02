#!/usr/bin/env python3
import argparse
import time

from SerialCommunicate import SerialCommunicate
from test_shutdown import ShutdownController


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
    parser.add_argument(
        "--send-lane",
        action="store_true",
        help="Send mode-0 lane command after connection",
    )
    parser.add_argument("--pos-x", type=float, default=0.0, help="Pose test pos_x")
    parser.add_argument("--pos-y", type=float, default=0.0, help="Pose test pos_y")
    parser.add_argument("--z-angle", type=float, default=0.0, help="Pose test z_angle")
    parser.add_argument("--speed-x", type=float, default=-0.28, help="Lane test speed_x")
    parser.add_argument("--deviation", type=float, default=0.0, help="Lane test deviation")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep sending the selected command repeatedly in a single process",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Loop send interval in seconds when --loop is enabled",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every received payload instead of only changed payloads",
    )
    return parser


def main():
    shutdown = ShutdownController("serial_test").install()
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
        if args.send_lane and args.send_pose:
            print("[serial_test] Please choose only one send mode: --send-lane or --send-pose")
            return 1

        if args.loop and not (args.send_lane or args.send_pose):
            print("[serial_test] --loop requires --send-lane or --send-pose")
            return 1

        sent_once = False
        next_send_time = 0.0

        last_payload = None
        while not shutdown.is_set():
            now = time.time()
            should_send = False
            if args.send_lane or args.send_pose:
                if args.loop:
                    should_send = now >= next_send_time
                else:
                    should_send = not sent_once

            if should_send:
                if args.send_lane:
                    uart.send_motion_mode(
                        0,
                        speed_x=args.speed_x,
                        deviation=args.deviation,
                    )
                else:
                    uart.send_motion_mode(
                        1,
                        pos_x=args.pos_x,
                        pos_y=args.pos_y,
                        z_angle=args.z_angle,
                    )

                sent_once = True
                if args.loop:
                    next_send_time = now + max(args.interval, 0.01)

            payload = uart.last_response
            if payload and (args.print_all or payload != last_payload):
                print(f"[serial_test] RX -> {payload}")
                last_payload = dict(payload)
            shutdown.wait(0.02)
    except KeyboardInterrupt:
        shutdown.request()
        return 0
    finally:
        uart.close()


if __name__ == "__main__":
    raise SystemExit(main())
