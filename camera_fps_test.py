import argparse
import time

import cv2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Test camera capture FPS on Jetson or Linux devices."
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
        "--duration",
        type=int,
        default=10,
        help="How long to measure FPS in seconds",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Warmup time before measuring in seconds",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable preview window",
    )
    return parser


def parse_device(device_arg):
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def draw_fps_overlay(frame, fps_value):
    overlay_text = f"FPS: {fps_value:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    text_size, baseline = cv2.getTextSize(overlay_text, font, scale, thickness)

    text_x = max(frame.shape[1] - text_size[0] - 16, 10)
    text_y = max(text_size[1] + 16, 20)

    cv2.rectangle(
        frame,
        (text_x - 10, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 10, text_y + baseline + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        overlay_text,
        (text_x, text_y),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def main():
    args = build_parser().parse_args()
    device = parse_device(args.device)
    show_display = not args.no_display

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 60)
    print("Camera FPS Test")
    print(f"Device: {args.device}")
    print(f"Requested: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Actual:    {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
    print(f"Warmup:    {args.warmup:.1f}s")
    print(f"Measure:   {args.duration}s")
    print(f"Display:   {'on' if show_display else 'off'}")
    print("Press q to quit preview early")
    print("=" * 60)

    if show_display:
        cv2.namedWindow("camera_fps_test", cv2.WINDOW_NORMAL)

    warmup_end = time.time() + args.warmup
    while time.time() < warmup_end:
        ok, _ = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed during warmup")

    frame_count = 0
    start_time = time.time()
    last_report_time = start_time
    last_report_count = 0
    instant_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed during measurement")
                break

            frame_count += 1
            now = time.time()
            elapsed = now - start_time
            report_elapsed = now - last_report_time

            if report_elapsed >= 1.0:
                instant_fps = (frame_count - last_report_count) / report_elapsed
                avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{elapsed:5.1f}s] instant_fps={instant_fps:6.2f} avg_fps={avg_fps:6.2f}"
                )
                last_report_time = now
                last_report_count = frame_count

            if show_display:
                draw_fps_overlay(frame, instant_fps)
                cv2.imshow("camera_fps_test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if elapsed >= args.duration:
                break
    finally:
        total_elapsed = max(time.time() - start_time, 1e-6)
        avg_fps = frame_count / total_elapsed
        print("=" * 60)
        print(f"Captured frames: {frame_count}")
        print(f"Measured time:   {total_elapsed:.2f}s")
        print(f"Average FPS:     {avg_fps:.2f}")
        print("=" * 60)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
