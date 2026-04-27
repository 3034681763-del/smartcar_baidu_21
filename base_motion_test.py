#!/usr/bin/env python3
import argparse
import threading
import time
from queue import Empty, Queue

from move_base import Base_func
from SerialCommunicate import SerialServer


DEFAULT_SEQUENCE = [
    ("Default", 0.3),
    ("moveshort", 0.6),
    ("backshort", 0.6),
    ("movelong", 0.8),
    ("backlong", 0.8),
    ("TurnLeftSmall", 0.6),
    ("TurnRightSmall", 0.6),
    ("TurnLeft", 0.8),
    ("TurnRight", 0.8),
    ("Default", 0.3),
]


def ack_print_loop(publish_queue, stop_event):
    while not stop_event.is_set():
        try:
            msg = publish_queue.get(timeout=0.1)
        except Empty:
            continue

        if msg.get("cmd") != "ActionDone":
            continue

        data = msg.get("data") or {}
        flag = data.get("flag")
        print(f"[base_motion_test] ACK received: flag={flag}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Standalone base motion group test."
    )
    parser.add_argument(
        "--physical-path",
        default="2.2:1.0",
        help="Serial physical path, for example 2.2:1.0 or /dev/ttyUSB0",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--actions",
        nargs="*",
        default=[],
        help="Optional custom action list. Example: moveshort TurnLeft backshort",
    )
    parser.add_argument(
        "--countdown",
        type=float,
        default=0.6,
        help="Default wait time after each action when --actions is used",
    )
    return parser


def main():
    args = build_parser().parse_args()
    request_queue = Queue()
    publish_queue = Queue()
    action_done_event = threading.Event()
    ack_stop_event = threading.Event()

    server = SerialServer(
        serial_path=args.physical_path,
        baudrate=args.baudrate,
        request_queue=request_queue,
        publish_queue=publish_queue,
        action_done_event=action_done_event,
    )
    server.run()
    ack_thread = threading.Thread(
        target=ack_print_loop,
        args=(publish_queue, ack_stop_event),
        daemon=True,
    )
    ack_thread.start()

    base = Base_func(
        request_queue=request_queue,
        use_lane_shm=False,
        action_done_event=action_done_event,
    )
    sequence = (
        [(action_key, args.countdown) for action_key in args.actions]
        if args.actions
        else DEFAULT_SEQUENCE
    )

    print("=" * 60)
    print("Base Motion Test")
    print(f"Serial path: {args.physical_path}")
    print("Sequence:")
    for action_key, countdown in sequence:
        print(f"  - {action_key} (wait {countdown:.2f}s)")
    print("=" * 60)

    try:
        for action_key, countdown in sequence:
            print(f"[base_motion_test] Running -> {action_key}")
            if not base.execute_base_motion(action_key, countdown=countdown):
                print(f"[base_motion_test] Unknown action: {action_key}")
                return 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[base_motion_test] Stopped by user.")
        return 0
    finally:
        ack_stop_event.set()
        base.MOD_STOP()
        time.sleep(0.1)
        server.close()

    print("[base_motion_test] Finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
