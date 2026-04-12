#!/usr/bin/env python3
import argparse
import time

import cv2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pure camera preview and FPS test for Jetson/Linux."
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Camera device index or path, e.g. 0 or /dev/video0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Requested camera FPS",
    )
    parser.add_argument(
        "--codec",
        default="MJPG",
        help="Requested FOURCC codec, e.g. MJPG or YUYV",
    )
    parser.add_argument(
        "--window-name",
        default="Pure Camera Test",
        help="Preview window name",
    )
    return parser


def parse_device(device_arg):
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def main():
    args = build_parser().parse_args()
    device = parse_device(args.device)

    print("=" * 50)
    print(" Pure Camera Preview And FPS Test ")
    print("=" * 50)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.device}")

    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters, e.g. MJPG")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Re-apply the requested size to avoid some drivers jumping back to larger modes.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_codec = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))

    print(f"Device:    {args.device}")
    print(f"Requested: {args.width}x{args.height} @ {args.fps} FPS, codec={args.codec}")
    print(f"Actual:    {actual_width}x{actual_height} @ {actual_fps:.2f} FPS, codec={actual_codec}")
    print("Preview:   on")
    print("Quit:      press q in the preview window")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    start_time = prev_time
    last_print_second = -1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read failed, device may have disconnected.")
                break

            frame_count += 1

            curr_time = time.time()
            diff = curr_time - prev_time
            if diff > 0:
                # Smooth the displayed FPS so the overlay is stable.
                fps = (fps * 0.9) + ((1.0 / diff) * 0.1)
            prev_time = curr_time

            elapsed = curr_time - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
            current_second = int(elapsed)
            if current_second != last_print_second and current_second >= 1:
                print(
                    f"[{elapsed:5.1f}s] instant_fps={fps:6.2f} avg_fps={avg_fps:6.2f}"
                )
                last_print_second = current_second

            cv2.rectangle(frame, (5, 5), (250, 70), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"AVG: {avg_fps:.1f}",
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(args.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exit requested by user.")
                break
    finally:
        total_elapsed = max(time.time() - start_time, 1e-6)
        final_avg_fps = frame_count / total_elapsed
        print("=" * 50)
        print(f"Captured frames: {frame_count}")
        print(f"Measured time:   {total_elapsed:.2f}s")
        print(f"Average FPS:     {final_avg_fps:.2f}")
        print("=" * 50)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
